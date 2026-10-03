"""Readable streamed Markdown, compact sources, and transient terminal reasoning."""

import json
import shutil
import sys
import time
import unicodedata
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from rich.text import Text

if __package__:
    from .citations import CitationFormatter
else:
    from citations import CitationFormatter


class _TailView:
    """Keep the newest lines visible when an unfinished block exceeds the screen."""

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False)
        height = max(2, console.size.height - 4)
        if len(lines) > height:
            yield Text("… showing latest lines", style="grey58")
            lines = lines[-(height - 1):]
        for line in lines:
            yield from line
            yield Segment.line()


class TerminalOutput:
    def __init__(self, stream=None, width=100, verbose=False):
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = self.stream.isatty()
        self.width = width
        self.verbose = verbose
        self.console = Console(
            file=self.stream, force_terminal=self.interactive,
            color_system="256" if self.interactive else None,
            width=width, markup=False, highlight=False,
        )
        self._formatted = self.interactive and self.console.is_interactive
        self.thinking_line = ""
        self.visible = False
        self.answer_started = False
        self._pending = ""
        self._live = None
        self._last_refresh = 0.0
        self._parser = MarkdownIt().enable("table")
        self._citations = CitationFormatter()

    def _sync_size(self):
        size = shutil.get_terminal_size(fallback=(80, 24))
        self.console.width = min(self.width, max(1, size.columns))
        self.console.height = max(1, size.lines)

    def _markdown(self, text):
        return Markdown(self._citations.format(text), justify="left", hyperlinks=False)

    def _start_live(self):
        if self._live is None:
            self._live = Live(
                Text(""), console=self.console, auto_refresh=False,
                transient=True, vertical_overflow="crop",
                redirect_stdout=False, redirect_stderr=False,
            )
            self._live.start(refresh=False)

    def _stop_live(self):
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()

    def _update_live(self):
        self._sync_size()
        renderable = self._markdown(self._pending)
        if self.visible:
            renderable = Group(renderable, Text(self._thinking_text(), style="grey50"))
        self._start_live()
        self._live.update(_TailView(renderable), refresh=True)
        self._last_refresh = time.monotonic()

    def _commit_completed(self):
        # Hold the last top-level Markdown block until a following block starts:
        # blank lines may still belong to a list, fenced code, or a table.
        blocks = [token for token in self._parser.parse(self._pending)
                  if token.level == 0 and token.map]
        if len(blocks) < 2:
            return
        lines = self._pending.splitlines(keepends=True)
        boundary = blocks[-1].map[0]
        completed = "".join(lines[:boundary])
        self._pending = "".join(lines[boundary:])
        self._stop_live()
        self.console.print(self._markdown(completed))

    def clear_thinking(self):
        live_thinking = self._live is not None and self.visible
        if self.visible and not live_thinking:
            self.stream.write("\r\033[2K\033[0m")
            self.stream.flush()
        self.visible = False
        self.thinking_line = ""
        if live_thinking:
            self._update_live()

    def _thinking_text(self):
        width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        fitted = []
        used = 0
        for char in reversed("Thinking: " + self.thinking_line):
            cells = 0 if unicodedata.combining(char) else (
                2 if unicodedata.east_asian_width(char) in ("W", "F") else 1)
            if used + cells > width:
                break
            fitted.append(char)
            used += cells
        return "".join(reversed(fitted))

    def thinking(self, text):
        if not self.interactive:
            return
        parts = text.replace("\r", "\n").split("\n")
        for index, part in enumerate(parts):
            if index:
                self.clear_thinking()
            self.thinking_line += "".join(char for char in part if char.isprintable())
            if not self.thinking_line:
                continue
            self.visible = True
            if self._live is not None:
                self._update_live()
            else:
                self.stream.write("\r\033[2K\033[38;5;244m" + self._thinking_text() + "\033[0m")
                self.stream.flush()

    def answer(self, text):
        self.clear_thinking()
        if not self._formatted:
            if not self.answer_started:
                self.stream.write("\nAssistant: ")
                self.answer_started = True
            self.stream.write(text)
            self.stream.flush()
            return
        if not self.answer_started:
            self._sync_size()
            self.console.print(Text("\nAssistant:", style="bold cyan"))
            self.answer_started = True
        self._pending += text
        # Repaint at most ten times per second rather than for every token.
        if self._live is None or time.monotonic() - self._last_refresh >= 0.1:
            self._commit_completed()
            self._update_live()

    def finish(self):
        self.clear_thinking()
        self._stop_live()
        if self.answer_started:
            if self._formatted:
                self._sync_size()
                if self._pending:
                    self.console.print(self._markdown(self._pending))
                if self._citations.sources:
                    self.console.print(Text("\nSources", style="bold grey63"))
                    for (path, pages), number in self._citations.sources.items():
                        self.console.print(Text(f"[{number}] {path}, {pages}", style="grey63"))
                self.console.print()
            else:
                self.stream.write("\n")
                self.stream.flush()
        self.answer_started = False
        self._pending = ""
        self._citations = CitationFormatter()

    def status(self, text, soft_wrap=False):
        self.clear_thinking()
        self._sync_size()
        self.console.print(Text(text, style="grey58"), soft_wrap=soft_wrap)

    def warning(self, text):
        self.clear_thinking()
        self._sync_size()
        self.console.print(Text("Warning: " + text, style="yellow"))

    def error(self, text):
        self.clear_thinking()
        self._sync_size()
        self.console.print(Text("Error: " + text, style="red"))

    def debug(self, text):
        if self.verbose:
            self.status(text, soft_wrap=True)

    def toggle_verbose(self, setting=None):
        self.verbose = not self.verbose if setting is None else setting
        self.status(f"Verbose output {'on' if self.verbose else 'off'}.")

    def prompt(self):
        if self.interactive:
            self.console.print(Text("\nYou: ", style="bold green"), end="")
            return input()
        return input("\nYou: ")

    def tool(self, name, arguments):
        if self.verbose:
            self.status(f"[Tool] {name}({json.dumps(arguments, ensure_ascii=False)})", soft_wrap=True)
            return
        filename = PurePosixPath(str(arguments.get("path", "PDF"))).name
        page = arguments.get("page", arguments.get("start_page", 1))
        end = arguments.get("end_page", page)
        pages = f"page {page}" if page == end else f"pages {page}–{end}"
        descriptions = {
            "list_directory": f"Listing {arguments.get('path', '.')}…",
            "list_pdfs": "Finding PDFs…",
            "search_collection": "Searching the PDF collection…",
            "search_pdf": f"Searching {filename}…",
            "read_pdf_pages": f"Reading {filename}, {pages}…",
            "read_pdf_content": f"Reading {filename}, {pages}…",
            "get_pdf_info": f"Inspecting {filename}…",
            "extract_pdf_tables": f"Extracting tables from {filename}, {pages}…",
            "ocr_pdf_pages": f"Recognizing text in {filename}, {pages}…",
            "calculate": "Calculating…",
        }
        self.status(descriptions.get(name, f"Running {name.replace('_', ' ')}…"))
