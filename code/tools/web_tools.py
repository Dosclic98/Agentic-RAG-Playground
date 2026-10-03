"""Optional, bounded web retrieval through Tavily's search and extract APIs."""

import ipaddress
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .tool_defs import ROOT


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # The authorization header belongs only to the fixed Tavily endpoint.
        return None


class WebTools:
    """Search public sources and cache extracted pages without uploading PDFs."""

    API_URL = "https://api.tavily.com"
    MAX_RESPONSE_BYTES = 4 * 1024 * 1024
    CACHE_SECONDS = 900
    CACHE_PAGES = 8

    def __init__(self, root: Path = ROOT, credentials_path: str = ".web_credentials.json",
                 timeout_seconds: int = 25):
        self.root = Path(root).resolve()
        self.credentials_path = credentials_path
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise ValueError("web_timeout_seconds must be between 1 and 60.")
        self.timeout_seconds = timeout_seconds
        self._page_cache = OrderedDict()

    @property
    def registry(self) -> dict:
        return {tool.__name__: tool for tool in (self.web_search, self.read_web_page)}

    def check_configuration(self):
        """Fail before starting an enabled chat if credentials are missing."""
        self._api_key()

    def _api_key(self):
        path = (self.root / self.credentials_path).resolve()
        if self.root not in path.parents:
            raise ValueError("Web credentials must stay inside the project.")
        try:
            with path.open("rb") as source:
                encoded = source.read(4097)
            if len(encoded) > 4096:
                raise ValueError
            credentials = json.loads(encoded)
            key = credentials.get("tavily_api_key") if isinstance(credentials, dict) else None
            if (not isinstance(key, str) or not key.strip()
                    or any(not 33 <= ord(c) <= 126 for c in key.strip())):
                raise ValueError
            return key.strip()
        except (OSError, ValueError):
            raise ValueError(
                "Web access needs a valid tavily_api_key in "
                f"{self.credentials_path}. Copy .web_credentials.example.json and fill in your key."
            ) from None

    @staticmethod
    def _bounded(value, maximum, name, minimum=1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")

    @staticmethod
    def _public_url(value):
        if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
            raise ValueError("Use a public HTTP or HTTPS URL of at most 2048 characters.")
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
            if (parsed.scheme not in ("http", "https") or not host
                    or parsed.username is not None or parsed.password is not None
                    or (port is not None and not 1 <= port <= 65535)):
                raise ValueError
            host = host.rstrip(".").lower()
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                if "." not in host or host.endswith((".localhost", ".local", ".internal")):
                    raise ValueError
            else:
                if not address.is_global:
                    raise ValueError
            return value
        except ValueError:
            raise ValueError("Use a public HTTP or HTTPS URL without login credentials.") from None

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _post(self, endpoint, payload):
        request = Request(
            self.API_URL + endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key(),
                     "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout_seconds) as response:
                encoded = response.read(self.MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            status = error.code
            error.close()
            explanations = {
                401: "Check your Tavily API key.", 403: "Access was denied.",
                429: "The search rate limit was reached; retry later.",
                432: "The Tavily usage limit was reached.",
                433: "The Tavily usage limit was reached.",
            }
            # Never echo an API response body: it may include credentials or queries.
            raise RuntimeError(f"Tavily HTTP {status}. " + explanations.get(
                status, "The provider could not complete the request.")) from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Could not reach Tavily within the configured timeout; check connectivity.") from None
        if len(encoded) > self.MAX_RESPONSE_BYTES:
            raise RuntimeError("Tavily returned a response exceeding the size limit.")
        try:
            result = json.loads(encoded)
        except (ValueError, UnicodeError):
            raise RuntimeError("Tavily returned an invalid JSON response.") from None
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise RuntimeError("Tavily returned an unexpected response format.")
        return result

    def web_search(self, query: str, max_results: Optional[int] = 5) -> str:
        """Find public web sources; read pages to verify claims beyond excerpts.

        Args:
            query: Short public search terms, at most 1000 characters.
            max_results: Number of sources, 1 to 10; defaults to 5.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("query must contain between 1 and 1000 characters.")
        max_results = 5 if max_results is None else max_results
        self._bounded(max_results, 10, "max_results")
        result = self._post("/search", {
            "query": query.strip(), "max_results": max_results, "search_depth": "basic",
            "include_answer": False, "include_raw_content": False,
            "include_images": False, "include_published_date": True,
        })
        sources = []
        for item in result["results"][:max_results]:
            if not isinstance(item, dict):
                continue
            try:
                url = self._public_url(item.get("url"))
            except ValueError:
                continue
            text = item.get("content") or ""
            if not isinstance(text, str):
                continue
            sources.append({
                "title": str(item.get("title") or "")[:300], "url": url,
                "published_date": str(item.get("published_date") or "")[:100] or None,
                "text": text[:2000], "truncated": len(text) > 2000,
            })
        return json.dumps({"source_type": "web", "provider": "tavily", "query": query.strip(),
                           "retrieved_at": self._timestamp(), "results": sources,
                           "returned_results": len(sources)}, ensure_ascii=False)

    def _page(self, url):
        cached = self._page_cache.get(url)
        if cached is not None and time.monotonic() - cached[0] < self.CACHE_SECONDS:
            self._page_cache.move_to_end(url)
            return cached[1]
        result = self._post("/extract", {
            "urls": [url], "extract_depth": "basic", "format": "text",
            "include_images": False, "timeout": min(self.timeout_seconds, 20),
        })
        pages = result["results"]
        if not pages or not isinstance(pages[0], dict):
            raise RuntimeError("Tavily returned no readable page text.")
        text = pages[0].get("raw_content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Tavily returned no readable page text.")
        page = {"url": self._public_url(pages[0].get("url") or url),
                "text": text, "retrieved_at": self._timestamp()}
        self._page_cache[url] = (time.monotonic(), page)
        self._page_cache.move_to_end(url)
        while len(self._page_cache) > self.CACHE_PAGES:
            self._page_cache.popitem(last=False)
        return page

    def read_web_page(self, url: str, offset: Optional[int] = 0,
                      max_chars: Optional[int] = 6000) -> str:
        """Read a public page. Follow next with the same max_chars for truncated text.

        Args:
            url: Public HTTP or HTTPS page URL.
            offset: Character position; defaults to 0. Use next.offset to continue.
            max_chars: Text budget, 1 to 12000 characters; defaults to 6000.
        """
        self._public_url(url)
        offset = 0 if offset is None else offset
        max_chars = 6000 if max_chars is None else max_chars
        self._bounded(max_chars, 12000, "max_chars")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        page = self._page(url)
        text = page["text"][offset:offset + max_chars]
        next_offset = offset + len(text)
        truncated = next_offset < len(page["text"])
        return json.dumps({
            "source_type": "web", "provider": "tavily", "requested_url": url,
            "url": page["url"], "retrieved_at": page["retrieved_at"], "offset": offset,
            "text": text, "total_chars": len(page["text"]), "truncated": truncated,
            "next": {"url": url, "offset": next_offset} if truncated else None,
        }, ensure_ascii=False)
