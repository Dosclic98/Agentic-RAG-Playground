"""Optional terminal editing with private, session-only question history."""

import sys
from contextlib import contextmanager


def _load_readline():
    # Importing readline changes input() globally, so load it only for real TTYs.
    try:
        import readline
    except ImportError:
        return None
    return readline


class TerminalInput:
    """Read questions with command completion and explicit multiline input.

    End a line with an unescaped backslash to continue on another line; end a
    line without one to submit. History stays on this instance, never in a file
    or the caller's global readline history.
    """

    COMMANDS = ("/exit", "/clear", "/verbose", "/verbose on", "/verbose off")

    def __init__(self, max_history=100):
        if type(max_history) is not int or max_history < 1:
            raise ValueError("max_history must be a positive integer.")
        self.max_history = max_history
        self._history = []

    def _complete(self, text, state):
        matches = [command for command in self.COMMANDS if command.startswith(text)]
        return matches[state] if 0 <= state < len(matches) else None

    @contextmanager
    def _editing(self, readline):
        previous_completer = readline.get_completer()
        previous_delimiters = readline.get_completer_delims()
        previous_history = [readline.get_history_item(index)
                            for index in range(1, readline.get_current_history_length() + 1)]
        try:
            readline.clear_history()
            for question in self._history:
                readline.add_history(question)
            # Keep spaces and '/' in the completion prefix for '/verbose on'.
            readline.set_completer_delims("\n")
            readline.set_completer(self._complete)
            if "libedit" in (readline.__doc__ or "").lower():
                readline.parse_and_bind("bind ^I rl_complete")
            else:
                readline.parse_and_bind("tab: complete")
                # A multiline paste remains one editable input until Enter.
                readline.parse_and_bind("set enable-bracketed-paste on")
            yield
        finally:
            readline.set_completer(previous_completer)
            readline.set_completer_delims(previous_delimiters)
            readline.clear_history()
            for question in previous_history:
                if question is not None:
                    readline.add_history(question)

    @staticmethod
    def _read_lines(prompt, continuation_prompt):
        lines = []
        while True:
            line = input(prompt if not lines else continuation_prompt)
            trailing_slashes = len(line) - len(line.rstrip("\\"))
            if trailing_slashes % 2:
                lines.append(line[:-1])
            else:
                lines.append(line)
                return "\n".join(lines)

    def read(self, prompt="", continuation_prompt="… "):
        """Return one complete question; EOF and interrupts propagate unchanged.

        ANSI prompts may use readline's \001 / \002 markers around invisible
        escape sequences so cursor positioning stays correct.
        """
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return input(prompt.replace("\001", "").replace("\002", ""))
        readline = _load_readline()
        if readline is None:
            question = self._read_lines(
                prompt.replace("\001", "").replace("\002", ""),
                continuation_prompt.replace("\001", "").replace("\002", ""),
            )
        else:
            with self._editing(readline):
                question = self._read_lines(prompt, continuation_prompt)
        if question.strip() and (not self._history or self._history[-1] != question):
            self._history.append(question)
            del self._history[:-self.max_history]
        return question
