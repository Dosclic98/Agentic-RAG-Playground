import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tools.web_tools import WebTools, _NoRedirect


class WebToolsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.key = "tvly-test-secret"
        self.credentials = self.root / ".web_credentials.json"
        self.credentials.write_text(json.dumps({"tavily_api_key": self.key}))
        self.tools = WebTools(self.root, timeout_seconds=17)
        self.url = "https://example.com/report"

    def response(self, payload):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(payload).encode("utf-8")
        return response

    def test_search_sends_only_query_and_returns_bounded_original_evidence(self):
        response = self.response({"answer": "Do not expose the provider's generated answer.",
            "results": [{"title": "Report", "url": self.url, "content": "é" * 2400,
                         "published_date": "2026-09-30"}]})
        with patch("tools.web_tools.build_opener") as factory:
            factory.return_value.open.return_value = response
            output = self.tools.web_search("  public company results  ", max_results=3)
        request = factory.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.tavily.com/search")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + self.key)
        payload = json.loads(request.data)
        self.assertEqual(payload["query"], "public company results")
        self.assertEqual(payload["max_results"], 3)
        self.assertFalse(payload["include_answer"])
        self.assertTrue(payload["include_published_date"])
        self.assertEqual(factory.return_value.open.call_args.kwargs["timeout"], 17)
        evidence = json.loads(output)
        self.assertEqual(evidence["source_type"], "web")
        self.assertEqual(evidence["results"][0]["published_date"], "2026-09-30")
        self.assertEqual(evidence["results"][0]["text"], "é" * 2000)
        self.assertTrue(evidence["results"][0]["truncated"])
        self.assertNotIn(self.key, output)
        self.assertNotIn("generated answer", output)

    def test_search_handles_no_matches_and_discards_invalid_sources(self):
        with patch.object(self.tools, "_post", return_value={"results": []}):
            self.assertEqual(json.loads(self.tools.web_search("nothing"))["results"], [])
        with patch.object(self.tools, "_post", return_value={"results": [
            {"url": "file:///etc/passwd", "content": "bad"}, None,
            {"url": self.url, "content": "usable"}]}):
            evidence = json.loads(self.tools.web_search("report"))
        self.assertEqual(evidence["returned_results"], 1)
        self.assertIsNone(evidence["results"][0]["published_date"])

    def test_invalid_arguments_make_no_network_requests(self):
        with patch.object(self.tools, "_post") as request:
            for query, limit in [("", 5), ("x" * 1001, 5), ([], 5), ("x", True),
                                 ("x", 0), ("x", 11)]:
                with self.subTest(query=type(query).__name__, limit=limit):
                    with self.assertRaises(ValueError):
                        self.tools.web_search(query, limit)
            for arguments in [dict(url=self.url, offset=-1), dict(url=self.url, offset=True),
                              dict(url=self.url, max_chars=0), dict(url=self.url, max_chars=12001)]:
                with self.assertRaises(ValueError):
                    self.tools.read_web_page(**arguments)
            request.assert_not_called()

    def test_nonpublic_urls_and_embedded_credentials_are_rejected(self):
        urls = ["file:///etc/passwd", "ftp://example.com/report", "http://localhost",
                "http://127.0.0.1", "http://10.0.0.5", "http://[::1]", "http://169.254.169.254",
                "https://service.internal", "https://host.local", "https://host.localhost",
                "https://user:password@example.com", "https://example.com:99999",
                "https://example.com/with space", "https://example.com/\n", "https:///missing-host"]
        with patch.object(self.tools, "_post") as request:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    self.tools.read_web_page(url)
            request.assert_not_called()

    def test_page_pagination_reuses_extraction_and_preserves_all_characters(self):
        text = "é first paragraph.\n" * 700
        response = self.response({"results": [{"url": self.url, "raw_content": text}], "failed_results": []})
        with patch("tools.web_tools.build_opener") as factory:
            factory.return_value.open.return_value = response
            arguments = {"url": self.url, "offset": 0}
            parts = []
            dates = set()
            while arguments is not None:
                evidence = json.loads(self.tools.read_web_page(**arguments))
                parts.append(evidence["text"])
                dates.add(evidence["retrieved_at"])
                self.assertEqual(evidence["truncated"], evidence["next"] is not None)
                arguments = evidence["next"]
            self.assertEqual(factory.return_value.open.call_count, 1)
            payload = json.loads(factory.return_value.open.call_args.args[0].data)
            self.assertEqual(payload["urls"], [self.url])
            self.assertEqual(payload["format"], "text")
            self.assertEqual(payload["timeout"], 17)
        self.assertEqual("".join(parts), text)
        self.assertEqual(len(dates), 1)

    def test_provider_redirect_url_is_cited_but_cursor_reuses_requested_url(self):
        resolved = "https://example.com/new-report"
        with patch.object(self.tools, "_post", return_value={"results": [
            {"url": resolved, "raw_content": "x" * 9000}]}):
            page = json.loads(self.tools.read_web_page(self.url))
        self.assertEqual(page["url"], resolved)
        self.assertEqual(page["next"], {"url": self.url, "offset": 6000})

    def test_cache_refreshes_after_expiry_and_evicts_old_pages(self):
        with patch.object(self.tools, "_post", return_value={"results": [
            {"url": self.url, "raw_content": "page text"}]}) as request:
            with patch("tools.web_tools.time.monotonic", return_value=0):
                self.tools.read_web_page(self.url)
            with patch("tools.web_tools.time.monotonic", return_value=901):
                self.tools.read_web_page(self.url)
                for index in range(8):
                    self.tools.read_web_page(f"https://example.com/page{index}")
                self.assertNotIn(self.url, self.tools._page_cache)
                self.assertEqual(len(self.tools._page_cache), 8)
            self.assertEqual(request.call_count, 10)

    def test_failed_extraction_is_reported_and_not_cached(self):
        with patch.object(self.tools, "_post", return_value={"results": [],
            "failed_results": [{"url": self.url, "error": "blocked"}]}):
            with self.assertRaisesRegex(RuntimeError, "no readable page text"):
                self.tools.read_web_page(self.url)
        self.assertFalse(self.tools._page_cache)

    def test_credentials_fail_with_actionable_messages_and_never_echo_secret_contents(self):
        for contents in ["not JSON " + self.key, "{}", "[]", '{"tavily_api_key": ""}',
                         json.dumps({"tavily_api_key": self.key + "\nBad: header"}), "x" * 5000]:
            self.credentials.write_text(contents)
            with self.subTest(contents=contents[:10]), self.assertRaises(ValueError) as error:
                self.tools.check_configuration()
            self.assertIn(".web_credentials.example.json", str(error.exception))
            self.assertNotIn(self.key, str(error.exception))
        self.credentials.unlink()
        with self.assertRaisesRegex(ValueError, "tavily_api_key"):
            self.tools.check_configuration()
        with self.assertRaisesRegex(ValueError, "inside the project"):
            WebTools(self.root, "../credentials.json").check_configuration()

    def test_provider_failures_are_sanitized_and_resources_are_closed(self):
        for status in (401, 429, 432, 500, 302):
            body = io.BytesIO(self.key.encode())
            failure = HTTPError("https://api.tavily.com/search", status, self.key, {}, body)
            with patch("tools.web_tools.build_opener") as factory:
                factory.return_value.open.side_effect = failure
                with self.assertRaises(RuntimeError) as error:
                    self.tools.web_search("public query")
            self.assertIn(str(status), str(error.exception))
            self.assertNotIn(self.key, str(error.exception))
            self.assertTrue(body.closed)
        with patch("tools.web_tools.build_opener") as factory:
            factory.return_value.open.side_effect = URLError(self.key)
            with self.assertRaisesRegex(RuntimeError, "connectivity") as error:
                self.tools.web_search("public query")
            self.assertNotIn(self.key, str(error.exception))

    def test_malformed_and_oversized_provider_responses_are_rejected(self):
        for encoded in (b"invalid", b"[]", b'{}', b'{"results": {}}', b'\xff',
                        b"x" * (self.tools.MAX_RESPONSE_BYTES + 1)):
            response = self.response({})
            response.read.return_value = encoded
            with patch("tools.web_tools.build_opener") as factory:
                factory.return_value.open.return_value = response
                with self.assertRaises(RuntimeError):
                    self.tools.web_search("public query")
            response.__exit__.assert_called_once()

    def test_authorization_is_not_forwarded_on_redirects(self):
        handler = _NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "Found", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
