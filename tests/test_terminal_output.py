import io
import itertools
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pyte
from ollama import ChatResponse, Message
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from terminal_output import TerminalOutput
from run_agent import PDFChatAgent


class FakeTerminal(io.StringIO):
    def isatty(self):
        return True


class TerminalOutputTests(unittest.TestCase):
    def setUp(self):
        terminal_env = patch.dict(os.environ, {"TERM": "xterm-256color", "TTY_INTERACTIVE": "1", "NO_COLOR": ""})
        terminal_env.start()
        self.addCleanup(terminal_env.stop)
        size = patch("terminal_output.shutil.get_terminal_size", return_value=os.terminal_size((80, 24)))
        size.start()
        self.addCleanup(size.stop)
        ticks = itertools.count()
        clock = patch("terminal_output.time.monotonic", side_effect=lambda: next(ticks))
        clock.start()
        self.addCleanup(clock.stop)

    @staticmethod
    def screen_text(value, columns=80, lines=24):
        screen = pyte.HistoryScreen(columns, lines, history=10000)
        # A real terminal's ONLCR output processing turns LF into CRLF.
        pyte.Stream(screen).feed(value.replace("\n", "\r\n"))
        history = ["".join(line[index].data for index in range(columns)) for line in screen.history.top]
        return "\n".join(history + screen.display), screen

    def test_new_thinking_line_replaces_previous_line(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.thinking("First")
        output.thinking(" thought\nSecond")
        self.assertEqual(output.thinking_line, "Second")
        self.assertIn("\033[38;5;244m", stream.getvalue())
        self.assertNotIn("\n", stream.getvalue())

    def test_answer_erases_thinking_and_resets_color(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.thinking("Thinking")
        output.answer("Answer")
        output.finish()
        self.assertFalse(output.visible)
        self.assertEqual(output.thinking_line, "")
        rendered, screen = self.screen_text(stream.getvalue())
        self.assertIn("Assistant:", rendered)
        self.assertIn("Answer", rendered)
        self.assertNotIn("Thinking: Thinking", rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_long_thinking_fits_one_terminal_row(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        with patch("terminal_output.shutil.get_terminal_size", return_value=type("Size", (), {"columns": 20})()):
            output.thinking("a" * 500 + "界" * 20)
        rendered = stream.getvalue().split("\033[38;5;244m")[-1].split("\033[0m")[0]
        self.assertLessEqual(len(rendered) * 2, 19)
        self.assertNotIn("\n", stream.getvalue())

    def test_redirected_output_omits_transient_trace_and_ansi(self):
        stream = io.StringIO()
        output = TerminalOutput(stream)
        output.thinking("Private temporary line")
        output.answer("Final answer")
        output.finish()
        self.assertEqual(stream.getvalue(), "\nAssistant: Final answer\n")

    def test_streamed_markdown_tables_and_citations_render_without_raw_markup(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        for chunk in ("## Revenue\n\n**Revenue grew.** [data/pdfs/report.", "pdf, p. 12]\n\n",
                      "| Year | USD millions |\n", "| --- | ---: |\n",
                      "| 2022 | 123.45 |\n", "| 2023 | 150.50 |\n\n", "A useful comparison."):
            output.answer(chunk)
        output.finish()
        rendered, _ = self.screen_text(stream.getvalue())
        self.assertNotIn("## Revenue", rendered)
        self.assertNotIn("**Revenue grew.**", rendered)
        self.assertNotIn("| --- |", rendered)
        self.assertIn("Revenue grew. [1]", rendered)
        self.assertIn("123.45", rendered)
        self.assertIn("150.50", rendered)
        self.assertEqual(rendered.count("data/pdfs/report.pdf, p. 12"), 1)

    def test_long_answers_remain_in_scrollback_without_duplicate_paragraphs(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        for index in range(60):
            output.answer(f"Unique paragraph {index:03d}: evidence and explanation.\n\n")
        output.finish()
        rendered, screen = self.screen_text(stream.getvalue())
        for index in range(60):
            self.assertEqual(rendered.count(f"Unique paragraph {index:03d}:"), 1)
        self.assertFalse(screen.cursor.hidden)

    def test_large_table_shows_latest_rows_then_keeps_every_row(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        text = "| Item | Value |\n| --- | ---: |\n"
        text += "".join(f"| item{index:03d} | {index} |\n" for index in range(70))
        output.answer(text)
        _, live_screen = self.screen_text(stream.getvalue())
        self.assertIn("item069", "\n".join(live_screen.display))
        output.finish()
        rendered, screen = self.screen_text(stream.getvalue())
        for index in range(70):
            self.assertEqual(rendered.count(f"item{index:03d}"), 1)
        self.assertNotIn("earlier lines", rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_blank_lines_inside_code_do_not_split_a_code_block(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        for chunk in ("```python\nfirst = 1\n\n", "second = 2\n", "```\n\nDone."):
            output.answer(chunk)
        output.finish()
        rendered, _ = self.screen_text(stream.getvalue())
        self.assertNotIn("```", rendered)
        self.assertEqual(rendered.count("first = 1"), 1)
        self.assertEqual(rendered.count("second = 2"), 1)

    def test_narrow_terminal_wraps_prose_without_dropping_words(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        with patch("terminal_output.shutil.get_terminal_size", return_value=os.terminal_size((35, 12))):
            output.answer("This paragraph explains revenue growth and preserves every word across narrow terminal lines.")
            output.finish()
        rendered, _ = self.screen_text(stream.getvalue(), columns=35, lines=12)
        flattened = " ".join(rendered.split())
        self.assertIn("revenue growth and preserves every word across narrow terminal lines", flattened)

    def test_thinking_after_answer_is_transient_and_finish_is_idempotent(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.answer("Partial answer")
        output.thinking("Interleaved thought\nNew thought")
        output.finish()
        finished = stream.getvalue()
        output.finish()
        self.assertEqual(stream.getvalue(), finished)
        rendered, screen = self.screen_text(finished)
        self.assertEqual(rendered.count("Partial answer"), 1)
        self.assertNotIn("New thought", rendered)
        self.assertNotIn("Interleaved thought", rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_progress_is_brief_and_verbose_reveals_arguments(self):
        stream = io.StringIO()
        output = TerminalOutput(stream)
        arguments = {"path": "data/pdfs/report.pdf", "page": 12, "offset": 100}
        output.tool("read_pdf_content", arguments)
        quiet = stream.getvalue()
        self.assertIn("Reading report.pdf, page 12", quiet)
        self.assertNotIn('"offset"', quiet)
        output.debug("generated tokens: 42")
        self.assertNotIn("generated tokens", stream.getvalue())
        output.toggle_verbose()
        output.tool("read_pdf_content", arguments)
        output.debug("generated tokens: 42")
        self.assertIn('"offset": 100', stream.getvalue())
        self.assertIn("generated tokens: 42", stream.getvalue())
        self.assertNotIn("\033", stream.getvalue())

    def test_interrupted_model_stream_leaves_readable_text_and_restores_cursor(self):
        def broken_stream():
            yield ChatResponse(message=Message(role="assistant", thinking="Transient thought"))
            yield ChatResponse(message=Message(role="assistant", content="**Partial answer**"))
            raise KeyboardInterrupt

        stream = FakeTerminal()
        agent = PDFChatAgent(client=Mock(chat=Mock(return_value=broken_stream())),
                             output_factory=lambda: TerminalOutput(stream))
        with self.assertRaises(KeyboardInterrupt):
            agent._stream_response([{"role": "user", "content": "Question"}])
        rendered, screen = self.screen_text(stream.getvalue())
        self.assertEqual(rendered.count("Partial answer"), 1)
        self.assertNotIn("Transient thought", rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_compact_citations_do_not_change_saved_model_messages(self):
        content = "Evidence [data/pdfs/report.pdf, p. 12]."
        response = ChatResponse(message=Message(role="assistant", content=content), done=True)
        stream = FakeTerminal()
        agent = PDFChatAgent(client=Mock(chat=Mock(return_value=iter([response]))),
                             output_factory=lambda: TerminalOutput(stream))
        agent.conversation.add_user("Question")
        agent.answer()
        self.assertEqual(agent.conversation.messages[-1].content, content)
        rendered, _ = self.screen_text(stream.getvalue())
        self.assertIn("Evidence [1]", rendered)
        self.assertIn("Sources", rendered)
        self.assertEqual(rendered.count("data/pdfs/report.pdf, p. 12"), 1)

    def test_web_sources_are_visible_and_distinct_from_pdf_sources(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.answer("Document [data/pdfs/report.pdf, p. 12]. External [https://example.com/report].")
        output.finish()
        rendered, _ = self.screen_text(stream.getvalue())
        self.assertIn("Document [1]. External [2].", rendered)
        self.assertIn("[1] data/pdfs/report.pdf, p. 12", rendered)
        self.assertIn("[2] Web: https://example.com/report", rendered)

    def test_startup_panel_shortens_home_path_and_fits_a_narrow_terminal(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        with patch("terminal_output.shutil.get_terminal_size", return_value=os.terminal_size((42, 24))):
            output.startup("qwen3.8:27b-q8_0", 131072, Path.home() / "Agentic-RAG-Playground", False)
        rendered, screen = self.screen_text(stream.getvalue(), columns=42)
        self.assertIn("Agentic PDF RAG", rendered)
        self.assertIn("qwen3.8:27b-q8_0", rendered)
        self.assertIn("131,072 tokens", rendered)
        self.assertIn("~/Agentic-RAG-Playground", rendered)
        self.assertIn("OFF", rendered)
        self.assertNotIn(str(Path.home()), rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_redirected_startup_and_tool_activity_stay_plain(self):
        stream = io.StringIO()
        output = TerminalOutput(stream)
        output.startup("local-model", 131072, "/tmp/project", True)
        output.begin_turn()
        output.activity("Waiting for the model…")
        output.tool("search_pdf", {"path": "data/pdfs/report.pdf"})
        output.answer("**Answer**")
        output.end_turn(generated_tokens=25)
        rendered = stream.getvalue()
        self.assertIn("Web access: on (Tavily)", rendered)
        self.assertIn("Searching report.pdf", rendered)
        self.assertIn("Assistant: **Answer**", rendered)
        self.assertNotIn("\033", rendered)
        self.assertNotIn("Waiting for", rendered)
        self.assertIsNone(output._live)

    def test_tool_activity_is_replaced_and_never_left_in_scrollback(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.begin_turn()
        output.tool("search_pdf", {"path": "data/pdfs/first.pdf"})
        output.tool("read_pdf_content", {"path": "data/pdfs/second.pdf", "page": 4})
        _, live_screen = self.screen_text(stream.getvalue())
        self.assertIn("Reading second.pdf, page 4", "\n".join(live_screen.display))
        self.assertNotIn("Searching first.pdf", "\n".join(live_screen.display))
        output.answer("The answer.")
        output.end_turn(generated_tokens=40)
        rendered, screen = self.screen_text(stream.getvalue())
        self.assertEqual(rendered.count("The answer."), 1)
        self.assertIn("2 tool calls", rendered)
        self.assertIn("40 generated tokens (turn total)", rendered)
        self.assertNotIn("Searching first.pdf", rendered)
        self.assertNotIn("Reading second.pdf", rendered)
        self.assertNotIn("Generating the answer", rendered)
        self.assertFalse(screen.cursor.hidden)
        self.assertIsNone(output._live)
        finished = stream.getvalue()
        output.end_turn(generated_tokens=40)
        self.assertEqual(stream.getvalue(), finished)

    def test_activity_footer_stays_visible_below_large_table_and_thinking(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.begin_turn()
        output.activity("Waiting for the model…")
        table = "| Item | Value |\n| --- | ---: |\n"
        table += "".join(f"| item{index:03d} | {index} |\n" for index in range(70))
        output.answer(table)
        output.thinking("Checking the figures")
        _, live_screen = self.screen_text(stream.getvalue())
        visible = "\n".join(live_screen.display)
        self.assertIn("item069", visible)
        self.assertIn("Checking the figures", visible)
        self.assertIn("Thinking…", visible)
        output.end_turn()
        rendered, screen = self.screen_text(stream.getvalue())
        for index in range(70):
            self.assertEqual(rendered.count(f"item{index:03d}"), 1)
        self.assertNotIn("Checking the figures", rendered)
        self.assertFalse(screen.cursor.hidden)

    def test_model_transport_failure_erases_activity_and_restores_cursor(self):
        stream = FakeTerminal()
        agent = PDFChatAgent(client=Mock(chat=Mock(side_effect=RuntimeError("unavailable"))),
                             output_factory=lambda: TerminalOutput(stream))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            agent.answer()
        rendered, screen = self.screen_text(stream.getvalue())
        self.assertNotIn("Waiting for the model", rendered)
        self.assertFalse(screen.cursor.hidden)
        self.assertIsNone(agent._get_output()._live)

    def test_fast_chunks_publish_latest_answer_before_stream_pauses(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        with patch("terminal_output.time.monotonic", return_value=10.0):
            output.activity("Waiting for the model…")
            output.answer("First chunk")
            output.answer(" and the newest chunk.")
            output._live.refresh()
            _, live_screen = self.screen_text(stream.getvalue())
            visible = "\n".join(live_screen.display)
            self.assertIn("First chunk and the newest chunk.", visible)
            self.assertIn("Generating the answer", visible)
        output.finish()

    def test_speculative_live_citation_is_not_committed_after_code_closes(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        output.activity("Waiting for the model…")
        output.answer("`Example [data/pdfs/report.pdf, p. 12]")
        output.answer("` is code.")
        output.finish()
        rendered, _ = self.screen_text(stream.getvalue())
        self.assertIn("data/pdfs/report.pdf, p. 12", rendered)
        self.assertNotIn("Sources", rendered)

    def test_very_short_terminals_preserve_answers_without_transient_lines(self):
        for height in (1, 2):
            with self.subTest(height=height), patch(
                    "terminal_output.shutil.get_terminal_size", return_value=os.terminal_size((35, height))):
                stream = FakeTerminal()
                output = TerminalOutput(stream)
                output.activity("Waiting for the model…")
                output.answer("First unique paragraph.\n\n")
                output.thinking("Ephemeral thought")
                output.answer("Second unique paragraph.\n\n")
                output.answer("Third unique paragraph.")
                output.finish()
                rendered, screen = self.screen_text(stream.getvalue(), columns=35, lines=height)
                for label in ("First", "Second", "Third"):
                    self.assertEqual(rendered.count(f"{label} unique paragraph."), 1)
                self.assertNotIn("Waiting for the model", rendered)
                self.assertNotIn("Ephemeral thought", rendered)
                self.assertFalse(screen.cursor.hidden)

    def test_blocking_tool_keeps_animating_and_interrupt_restores_cursor(self):
        stream = FakeTerminal()
        output = TerminalOutput(stream)
        entered, release = threading.Event(), threading.Event()

        def blocking_tool():
            entered.set()
            release.wait(2)
            raise KeyboardInterrupt

        call = Message.ToolCall(function=Message.ToolCall.Function(name="blocking_tool", arguments={}))
        client = Mock(chat=Mock(return_value=iter([ChatResponse(
            message=Message(role="assistant", tool_calls=[call]), done=True)])))
        agent = PDFChatAgent(client=client, output_factory=lambda: output)
        agent.tools["blocking_tool"] = blocking_tool
        interrupted = []

        def run():
            try:
                agent.answer()
            except KeyboardInterrupt:
                interrupted.append(True)

        # Use a real clock for the only test that exercises the refresh thread.
        with patch("terminal_output.time.monotonic", time.perf_counter):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                before = len(stream.getvalue())
                deadline = time.perf_counter() + 1
                while len(stream.getvalue()) <= before and time.perf_counter() < deadline:
                    release.wait(0.02)
                self.assertGreater(len(stream.getvalue()), before)
            finally:
                release.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(interrupted, [True])
        rendered, screen = self.screen_text(stream.getvalue())
        self.assertNotIn("Running blocking tool", rendered)
        self.assertNotIn("Preparing the next response", rendered)
        self.assertFalse(screen.cursor.hidden)
        self.assertIsNone(output._live)


if __name__ == "__main__":
    unittest.main()
