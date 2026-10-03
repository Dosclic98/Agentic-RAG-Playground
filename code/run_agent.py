"""Streaming terminal chat with a local Ollama model and PDF retrieval tools."""

import json
from ollama import Client, Message, ResponseError

if __package__:
    from .config import AgentConfig
    from .conversation import Conversation
    from .prompts import SYSTEM_PROMPT
    from .tools import PDFTools
    from .terminal_output import TerminalOutput
else:
    from config import AgentConfig
    from conversation import Conversation
    from prompts import SYSTEM_PROMPT
    from tools import PDFTools
    from terminal_output import TerminalOutput


class PDFChatAgent:
    """Own the model client, tools, conversation, and terminal chat loop."""

    def __init__(self, config=None, client=None, pdf_tools=None, output_factory=None):
        self.config = config if config is not None else AgentConfig()
        self.client = client if client is not None else Client(host=self.config.host)
        self.pdf_tools = (
            pdf_tools if pdf_tools is not None else PDFTools(
                self.config.project_root, ocr_data_path=self.config.ocr_data_path)
        )
        self.tools = self.pdf_tools.registry
        self.output_factory = output_factory if output_factory is not None else lambda: TerminalOutput(
            width=self.config.output_width, verbose=self.config.verbose)
        self._output = None
        self.conversation = Conversation(
            SYSTEM_PROMPT, self.config.context_length, self.config.max_output_tokens)

    def _get_output(self):
        if self._output is None:
            self._output = self.output_factory()
        return self._output

    def _stream_response(self, history):
        output = self._get_output()
        thinking = []
        content = []
        tool_calls = []
        last_chunk = None
        stream = self.client.chat(
            model=self.config.model,
            messages=history,
            tools=list(self.tools.values()),
            stream=True,
            think=True,
            options={
                "num_ctx": self.config.context_length,
                "num_predict": self.config.max_output_tokens,
            },
        )
        try:
            for chunk in stream:
                last_chunk = chunk
                if chunk.message.thinking:
                    thinking.append(chunk.message.thinking)
                    output.thinking(chunk.message.thinking)
                if chunk.message.content:
                    content.append(chunk.message.content)
                    output.answer(chunk.message.content)
                if chunk.message.tool_calls:
                    output.clear_thinking()
                    tool_calls.extend(chunk.message.tool_calls)
        finally:
            output.finish()
            close = getattr(stream, "close", None)
            if close:
                close()
        if last_chunk is None:
            raise RuntimeError("Ollama returned an empty response stream.")
        if not last_chunk.done:
            raise RuntimeError("Ollama's response stream ended before completion.")
        return last_chunk.model_copy(update={"message": Message(
            role="assistant", content="".join(content), thinking="".join(thinking),
            tool_calls=tool_calls or None,
        )})

    def answer(self):
        """Answer the current question, retrieving evidence until the model finishes."""
        # Continue tool use until an answer is returned or the user interrupts.
        compatibility_mode = False
        while True:
            try:
                response = self._request_response(compatibility_mode)
            except ResponseError as error:
                if (compatibility_mode or error.status_code != 500
                        or "no user query found" not in error.error.lower()):
                    raise
                self._get_output().status("Retrying the response…")
                self._get_output().debug("[Chat] Using a compact conversation for Ollama compatibility.")
                compatibility_mode = True
                response = self._request_response(True)

            message = response.message
            self.conversation.messages.append(message)

            if not message.tool_calls:
                reason = getattr(response, "done_reason", None) or "unknown"
                tokens = getattr(response, "eval_count", None)
                self._get_output().debug(f"[Chat] Generation ended: {reason}; generated tokens: {tokens}.")
                if reason == "length":
                    self._get_output().warning("The response reached the generation limit and may be "
                                               "incomplete. Ask the model to continue, or narrow the request.")
                elif not message.content:
                    self._get_output().warning("The model returned no answer and no tool calls. "
                                               "Try reformulating the question.")
                return

            for call in message.tool_calls:
                self._execute_tool(call)

    def _request_response(self, compatibility_mode):
        history = (self.conversation.compatibility_messages() if compatibility_mode
                   else self.conversation.messages)
        return self._stream_response(history)

    def _execute_tool(self, call):
        """Dispatch one tool call and save its evidence or error in the conversation."""
        name = call.function.name
        arguments = call.function.arguments
        self._get_output().tool(name, arguments)
        try:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            result = self.tools[name](**arguments)
        except Exception as error:
            result = json.dumps({"error": str(error)})
            self._get_output().error(f"{name}: {error}")
        self.conversation.messages.append({
            "role": "tool", "tool_name": name, "content": result,
        })

    def run(self):
        """Read terminal input and handle commands, failures, and interrupts."""
        output = self._get_output()
        output.status(f"Model: {self.config.model}")
        output.status(f"Context window: {self.config.context_length:,} tokens")
        output.status(f"Project: {self.pdf_tools.root}")
        output.status("Commands: /exit to quit, /clear to reset, /verbose to toggle tool details.")

        while True:
            try:
                prompt = output.prompt().strip()

                if prompt == "/exit":
                    break
                if prompt == "/clear":
                    self.conversation.clear()
                    output.status("Conversation cleared.")
                    continue
                if prompt == "/verbose" or prompt.startswith("/verbose "):
                    setting = prompt[len("/verbose"):].strip().lower()
                    if setting not in ("", "on", "off"):
                        output.warning("Use /verbose, /verbose on, or /verbose off.")
                    else:
                        output.toggle_verbose(None if not setting else setting == "on")
                    continue
                if not prompt:
                    continue

                checkpoint = self.conversation.add_user(prompt)

                try:
                    self.answer()
                except Exception as error:
                    self.conversation.recover_failed_turn(checkpoint)
                    output.error(str(error))

            except (KeyboardInterrupt, EOFError):
                output.status("Goodbye.")
                break


def run():
    """Create an independent agent and start the terminal chat."""
    PDFChatAgent().run()


if __name__ == "__main__":
    run()
