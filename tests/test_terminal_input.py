import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from terminal_input import TerminalInput


class FakeReadline:
    """Model readline's automatic input history and replaceable global state."""

    __doc__ = "GNU readline"

    def __init__(self):
        self.completer = object()
        self.delimiters = "original delimiters"
        self.history = ["existing host history"]
        self.bindings = []

    def get_completer(self):
        return self.completer

    def set_completer(self, value):
        self.completer = value

    def get_completer_delims(self):
        return self.delimiters

    def set_completer_delims(self, value):
        self.delimiters = value

    def get_current_history_length(self):
        return len(self.history)

    def get_history_item(self, index):
        return self.history[index - 1]

    def clear_history(self):
        self.history.clear()

    def add_history(self, value):
        self.history.append(value)

    def parse_and_bind(self, value):
        self.bindings.append(value)


class TerminalInputTests(unittest.TestCase):
    def setUp(self):
        self.readline = FakeReadline()
        self.initial_completer = self.readline.completer
        self.terminal_input = TerminalInput(max_history=2)
        for name in ("stdin", "stdout"):
            terminal = patch("terminal_input.sys." + name, Mock(isatty=Mock(return_value=True)))
            terminal.start()
            self.addCleanup(terminal.stop)
        module = patch("terminal_input._load_readline", return_value=self.readline)
        self.loader = module.start()
        self.addCleanup(module.stop)

    def assert_restored(self):
        self.assertIs(self.readline.completer, self.initial_completer)
        self.assertEqual(self.readline.delimiters, "original delimiters")
        self.assertEqual(self.readline.history, ["existing host history"])

    def test_command_completion_includes_verbose_arguments(self):
        def read(prompt):
            completer = self.readline.completer
            self.assertEqual(self.readline.delimiters, "\n")
            self.assertEqual(completer("/verb", 0), "/verbose")
            self.assertEqual(completer("/verbose o", 0), "/verbose on")
            self.assertEqual(completer("/verbose o", 1), "/verbose off")
            self.assertIsNone(completer("/verbose o", 2))
            self.assertIsNone(completer("ordinary question", 0))
            return "/verbose on"
        with patch("builtins.input", side_effect=read):
            self.assertEqual(self.terminal_input.read("You: "), "/verbose on")
        self.assertIn("tab: complete", self.readline.bindings)
        self.assertIn("set enable-bracketed-paste on", self.readline.bindings)
        self.assert_restored()

    def test_session_history_is_bounded_and_global_private_text_is_removed(self):
        expected_histories = [[], ["private first"], ["private first", "second"],
                              ["second", "third"]]
        for answer, expected in zip(("private first", "second", "third", "third"), expected_histories):
            def read(prompt):
                self.assertEqual(self.readline.history, expected)
                # Simulate input()'s automatic addition without touching its setting.
                self.readline.add_history(answer)
                return answer
            with patch("builtins.input", side_effect=read):
                self.assertEqual(self.terminal_input.read(), answer)
            self.assert_restored()
        self.assertEqual(self.terminal_input._history, ["second", "third"])

    def test_multiline_is_one_question_and_one_history_entry(self):
        with patch("builtins.input", side_effect=["Compare the reports\\", "and cite pages\\", "please."]) as read:
            question = self.terminal_input.read("You: ", continuation_prompt="… ")
        self.assertEqual(question, "Compare the reports\nand cite pages\nplease.")
        self.assertEqual([call.args[0] for call in read.call_args_list], ["You: ", "… ", "… "])
        self.assertEqual(self.terminal_input._history, [question])
        self.assert_restored()

    def test_escaped_backslash_submits_without_continuation(self):
        value = "literal " + "\\" * 2
        with patch("builtins.input", return_value=value) as read:
            self.assertEqual(self.terminal_input.read(), value)
        read.assert_called_once()
        self.assert_restored()

    def test_pasted_embedded_newlines_are_returned_in_one_question(self):
        with patch("builtins.input", return_value="First paragraph.\nSecond paragraph.") as read:
            self.assertEqual(self.terminal_input.read(), "First paragraph.\nSecond paragraph.")
        read.assert_called_once()
        self.assert_restored()

    def test_interrupts_and_eof_restore_settings_and_do_not_save_partial_queries(self):
        for error in (KeyboardInterrupt, EOFError):
            with self.subTest(error=error.__name__):
                with patch("builtins.input", side_effect=["private partial\\", error()]):
                    with self.assertRaises(error):
                        self.terminal_input.read()
                self.assertFalse(self.terminal_input._history)
                self.assert_restored()

    def test_missing_readline_still_supports_explicit_multiline(self):
        self.loader.return_value = None
        with patch("builtins.input", side_effect=["first\\", "second"]):
            self.assertEqual(self.terminal_input.read(), "first\nsecond")
        self.assert_restored()

    def test_missing_readline_strips_prompt_markers_but_preserves_trusted_color(self):
        self.loader.return_value = None
        prompt = "\001\033[32m\002You › \001\033[0m\002"
        continuation = "\001\033[90m\002… \001\033[0m\002"
        with patch("builtins.input", side_effect=["first\\", "second"]) as read:
            self.assertEqual(self.terminal_input.read(prompt, continuation), "first\nsecond")
        self.assertEqual([call.args[0] for call in read.call_args_list],
                         ["\033[32mYou › \033[0m", "\033[90m… \033[0m"])

    def test_redirected_input_or_output_uses_plain_input(self):
        for name in ("stdin", "stdout"):
            with self.subTest(stream=name):
                with patch("terminal_input.sys." + name, Mock(isatty=Mock(return_value=False))):
                    with patch("builtins.input", return_value="literal backslash\\") as read:
                        self.assertEqual(self.terminal_input.read("Plain: "), "literal backslash\\")
                    read.assert_called_once_with("Plain: ")
        self.loader.assert_not_called()
        self.assertFalse(self.terminal_input._history)

    def test_redirected_input_strips_readline_prompt_markers(self):
        with patch("terminal_input.sys.stdin", Mock(isatty=Mock(return_value=False))):
            with patch("builtins.input", return_value="question") as read:
                self.assertEqual(self.terminal_input.read("\001You › \002"), "question")
        read.assert_called_once_with("You › ")
        self.loader.assert_not_called()

    def test_libedit_uses_compatible_completion_binding(self):
        self.readline.__doc__ = "libedit readline"
        with patch("builtins.input", return_value="/exit"):
            self.terminal_input.read()
        self.assertEqual(self.readline.bindings, ["bind ^I rl_complete"])
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
