"""Compact full PDF citations for display while preserving their source details."""

import re

from markdown_it import MarkdownIt


class CitationFormatter:
    PATTERN = re.compile(
        r"\[(?P<path>[^\]\n]+?\.pdf)\s*,\s*"
        r"(?P<pages>pp?\.\s*\d+(?:\s*(?:[-–,]|and)\s*\d+)*)\]",
        re.IGNORECASE,
    )
    INLINE_CODE = re.compile(r"(`+).*?\1", re.DOTALL)

    def __init__(self):
        self.sources = {}
        self.parser = MarkdownIt()

    def _replace(self, text):
        def reference(match):
            path = match.group("path").strip()
            pages = re.sub(r"\s+", " ", match.group("pages").strip())
            pages = re.sub(r"^pp?\.\s*", lambda label: label.group().strip().lower() + " ", pages)
            source = (path, pages)
            if source not in self.sources:
                self.sources[source] = len(self.sources) + 1
            return f"[{self.sources[source]}]"

        pieces = []
        position = 0
        for match in self.INLINE_CODE.finditer(text):
            pieces.append(self.PATTERN.sub(reference, text[position:match.start()]))
            pieces.append(match.group())
            position = match.end()
        pieces.append(self.PATTERN.sub(reference, text[position:]))
        return "".join(pieces)

    def format(self, markdown):
        lines = markdown.splitlines(keepends=True)
        pieces = []
        position = 0
        for token in self.parser.parse(markdown):
            if token.type in ("fence", "code_block") and token.map:
                start, end = token.map
                pieces.append(self._replace("".join(lines[position:start])))
                pieces.append("".join(lines[start:end]))
                position = end
        pieces.append(self._replace("".join(lines[position:])))
        return "".join(pieces)
