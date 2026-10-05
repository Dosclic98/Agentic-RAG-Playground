"""Readable streamed Markdown, compact sources, and transient terminal reasoning."""

import json
import shutil
import sys
import time
import unicodedata
from pathlib import Path, PurePosixPath

from markdown_it import MarkdownIt
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if __package__:
    from .citations import CitationFormatter
    from .generation_metrics import GenerationMetrics
    from .terminal_input import TerminalInput
else:
    from citations import CitationFormatter
    from generation_metrics import GenerationMetrics
    from terminal_input import TerminalInput


class _TailView:
    """Keep the newest lines visible when an unfinished block exceeds the screen."""

    def __init__(self, renderable, footer=()):
        self.renderable = renderable
        self.footer = footer

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False) if self.renderable else []
        height = max(1, console.size.height - 4)
        footer_lines = [line for item in self.footer
                        for line in console.render_lines(item, options, pad=False)]
        available = max(0, height - len(footer_lines))
        displayed = []
        if len(lines) > available:
            if available > 1:
                displayed += console.render_lines(Text("… showing latest lines", style="grey58"),
                                                  options, pad=False)
                available -= 1
            lines = lines[-available:] if available else []
        displayed += lines + footer_lines[-height:]
        for index, line in enumerate(displayed):
            yield from line
            if index < len(displayed) - 1:
                yield Segment.line()


class _ActivityLine:
    """Animate a bounded status line, including while the caller is blocked."""

    def __init__(self, label, started, show_elapsed=True):
        self.label = label
        self.started = started
        self.show_elapsed = show_elapsed
        self.spinner = Spinner("dots", style="cyan")

    def __rich_console__(self, console, options):
        frame = self.spinner.render(console.get_time())
        elapsed = max(0.0, time.monotonic() - self.started) if self.show_elapsed else 0.0
        timer = Text(f" · {elapsed:.1f}s" if self.show_elapsed else "", style="grey50")
        label = Text(self.label, style="grey63")
        label.truncate(max(0, options.max_width - frame.cell_len - timer.cell_len - 1),
                       overflow="ellipsis")
        line = Text.assemble(frame, " ", label, timer)
        line.no_wrap = True
        line.overflow = "ellipsis"
        yield line


def _metrics_text(snapshot, compact=False, average=False):
    prefix = "~" if snapshot.estimated else ""
    unit = "tok" if compact else "tokens"
    rate_unit = "avg t/s" if average else "t/s"
    if snapshot.rate is None:
        rate = f"— {rate_unit}"
    else:
        rate_prefix = "~" if snapshot.rate_estimated else ""
        rate = f"{rate_prefix}{snapshot.rate:.1f} {rate_unit}"
    return (f"{prefix}{snapshot.tokens:,} {unit} · {snapshot.elapsed:.1f}s · "
            f"{rate}")


class _MetricsLine:
    """Read a consistent snapshot each refresh, including during tool waits."""

    def __init__(self, metrics):
        self.metrics = metrics

    def __rich_console__(self, console, options):
        yield Text(_metrics_text(self.metrics.snapshot(), compact=options.max_width < 50),
                   style="grey50")


class _MarkdownPreview:
    """Parse only the latest published snapshot, with provisional citations."""

    def __init__(self, text, sources):
        self.text = text
        self.sources = dict(sources)
        self._rendered = None

    def __rich_console__(self, console, options):
        if self._rendered is None:
            formatter = CitationFormatter()
            formatter.sources.update(self.sources)
            self._rendered = Markdown(formatter.format(self.text), justify="left", hyperlinks=False)
        yield self._rendered


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
        self._activity = None
        self._turn_started = None
        self._tool_count = 0
        self._metrics = GenerationMetrics()
        self._input = TerminalInput()
        self._parser = MarkdownIt().enable("table")
        self._citations = CitationFormatter()

    def _sync_size(self):
        size = shutil.get_terminal_size(fallback=(80, 24))
        self.console.width = min(self.width, max(1, size.columns))
        self.console.height = max(1, size.lines)

    def _markdown(self, text):
        return Markdown(self._citations.format(text), justify="left", hyperlinks=False)

    def _start_live(self):
        animated = self._activity is not None or self._turn_started is not None
        if self._live is not None and self._live.auto_refresh != animated:
            # Rich starts/stops its refresh thread with the Live instance.
            self._stop_live()
        if self._live is None:
            self._live = Live(
                Text(""), console=self.console, auto_refresh=animated,
                refresh_per_second=8,
                transient=True, vertical_overflow="crop",
                redirect_stdout=False, redirect_stderr=False,
            )
            self._live.start(refresh=False)

    def _stop_live(self):
        if self._live is not None:
            live, self._live = self._live, None
            live.stop()

    def _update_live(self, refresh=True):
        self._sync_size()
        # Rich's final newline can scroll a transient line off a one-row screen.
        if self.console.height < 2:
            self._stop_live()
            return
        renderable = _MarkdownPreview(self._pending, self._citations.sources) if self._pending else None
        footer = []
        if self.visible:
            footer.append(Text(self._thinking_text(), style="grey50", no_wrap=True,
                               overflow="ellipsis"))
        if self._activity is not None:
            footer.append(self._activity)
        if self._turn_started is not None:
            footer.append(_MetricsLine(self._metrics))
        self._start_live()
        self._live.update(_TailView(renderable, footer), refresh=refresh)
        if refresh:
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
        self._phase("Thinking…")
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
        self._phase("Generating the answer…")
        if not self._formatted:
            if not self.answer_started:
                self.stream.write("\nAssistant: ")
                self.answer_started = True
            self.stream.write(text)
            self.stream.flush()
            return
        if not self.answer_started:
            self._stop_live()
            self._sync_size()
            self.console.print()
            self.console.rule(Text("Assistant:", style="bold cyan"),
                              align="left", style="grey35")
            self.answer_started = True
        self._pending += text
        # Repaint at most ten times per second rather than for every token.
        refresh = self._live is None or time.monotonic() - self._last_refresh >= 0.1
        if refresh:
            self._commit_completed()
        # Publish every chunk; the refresh thread can show it during a pause.
        # Markdown parsing is deferred until the snapshot is actually rendered.
        self._update_live(refresh=refresh)

    def finish(self):
        if self._live is None:
            self.clear_thinking()
        else:
            self.visible = False
            self.thinking_line = ""
        self._activity = None
        self._stop_live()
        if self.answer_started:
            if self._formatted:
                self._sync_size()
                if self._pending:
                    self.console.print(self._markdown(self._pending))
                if self._citations.sources:
                    self.console.print(Text("\nSources", style="bold grey63"))
                    for (path, pages), number in self._citations.sources.items():
                        label = f"Web: {path}" if pages == "web" else f"{path}, {pages}"
                        self.console.print(Text(f"[{number}] {label}", style="grey63"))
                self.console.print()
            else:
                self.stream.write("\n")
                self.stream.flush()
        self.answer_started = False
        self._pending = ""
        self._citations = CitationFormatter()

    def status(self, text, soft_wrap=False):
        self.finish()
        self._sync_size()
        self.console.print(Text(text, style="grey58"), soft_wrap=soft_wrap)

    def warning(self, text):
        self.finish()
        self._sync_size()
        self.console.print(Text("Warning: " + text, style="yellow"))

    def error(self, text):
        self.finish()
        self._sync_size()
        self.console.print(Text("Error: " + text, style="red"))

    def debug(self, text):
        if self.verbose:
            self.status(text, soft_wrap=True)

    def toggle_verbose(self, setting=None):
        self.verbose = not self.verbose if setting is None else setting
        self.status(f"Verbose output {'on' if self.verbose else 'off'}.")

    def prompt(self):
        self.finish()
        if self._formatted:
            if sys.stdin.isatty() and sys.stdout.isatty():
                self.console.print()
                if self.console.no_color:
                    return self._input.read("You › ")
                # Readline needs the prompt width to position/wrap editable text.
                return self._input.read(
                    "\001\033[1;32m\002You › \001\033[0m\002",
                    continuation_prompt="\001\033[38;5;244m\002… \001\033[0m\002")
            self.console.print(Text("\nYou › ", style="bold green"), end="")
            return input()
        return input("\nYou: ")

    def startup(self, model, context_length, project_root, web_enabled):
        """Show concise settings without changing the plain-output fallback."""
        if not self._formatted:
            for line in (f"Model: {model}", f"Context window: {context_length:,} tokens",
                         f"Project: {project_root}",
                         f"Web access: {'on (Tavily)' if web_enabled else 'off'}",
                         "Commands: /exit to quit, /clear to reset, /verbose to toggle tool details."):
                self.status(line)
            return
        self.finish()
        self._sync_size()
        project = Path(project_root)
        try:
            project_label = "~/" + str(project.relative_to(Path.home()))
        except ValueError:
            project_label = str(project)
        settings = Table.grid(padding=(0, 2))
        settings.add_column(style="grey63", no_wrap=True)
        settings.add_column(overflow="fold")
        settings.add_row("Model", Text(str(model)))
        settings.add_row("Context", Text(f"{context_length:,} tokens"))
        settings.add_row("Project", Text(project_label, style="grey63"))
        settings.add_row("Web", Text("ON · Tavily" if web_enabled else "OFF",
                                     style="green" if web_enabled else "grey58"))
        self.console.print(Panel(settings, title=Text("Agentic PDF RAG", style="bold cyan"),
                                 title_align="left", border_style="grey35", padding=(0, 1)))
        self.console.print(Text(" /exit  quit   /clear  reset   /verbose  tool details", style="grey58"))
        self.console.print(Text(" Tab  commands   ↑/↓  history   \\ + Enter  multiline", style="grey50"))

    def begin_turn(self):
        self.finish()
        self._turn_started = time.monotonic()
        self._tool_count = 0
        self._metrics.begin_turn(started=self._turn_started)

    def begin_generation(self):
        if self._turn_started is not None:
            self._metrics.begin_generation()

    def generation_progress(self, chunk):
        if self._turn_started is not None:
            self._metrics.progress(chunk)

    def end_generation(self):
        if self._turn_started is not None:
            self._metrics.end_generation()

    def end_turn(self, generated_tokens=None):
        self.finish()
        if self._turn_started is None:
            return
        self._metrics.finish_turn(generated_tokens=generated_tokens)
        snapshot = self._metrics.snapshot()
        self._turn_started = None
        if self._formatted:
            details = ["Turn total: " + _metrics_text(snapshot, average=True),
                       f"{self._tool_count} tool call{'s' if self._tool_count != 1 else ''}"]
            self._sync_size()
            self.console.print(Text(" · ".join(details), style="grey50"))

    def _phase(self, label):
        if self._activity is not None and self._activity.label != label:
            self._activity = _ActivityLine(label, self._activity.started, self._activity.show_elapsed)

    def activity(self, text):
        if not self._formatted:
            return
        self.clear_thinking()
        # Labels are a single plain-text line, even for unusual filenames.
        label = " ".join("".join(c for c in str(text) if c.isprintable() or c.isspace()).split())
        if self._activity is None or self._activity.label != label:
            self._activity = _ActivityLine(label, time.monotonic(), self._turn_started is None)
        self._update_live()

    def tool(self, name, arguments):
        if self._turn_started is not None:
            self._tool_count += 1
        if self.verbose:
            self.status(f"[Tool] {name}({json.dumps(arguments, ensure_ascii=False)})", soft_wrap=True)
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
            "web_search": "Searching the web…",
            "read_web_page": "Reading a web page…",
        }
        description = descriptions.get(name, f"Running {name.replace('_', ' ')}…")
        if self._formatted:
            self.activity(description)
        elif not self.verbose:
            self.status(description)
