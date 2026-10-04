"""Streaming terminal chat with a local Ollama model and PDF retrieval tools."""

import json
from ollama import Client, Message, ResponseError

if __package__:
    from .config import AgentConfig
    from .conversation import Conversation
    from .prompts import SYSTEM_PROMPT, build_system_prompt
    from .tools import PDFTools, WebTools
    from .terminal_output import TerminalOutput
else:
    from config import AgentConfig
    from conversation import Conversation
    from prompts import SYSTEM_PROMPT, build_system_prompt
    from tools import PDFTools, WebTools
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
        self.tools = dict(self.pdf_tools.registry)
        self.web_tools = None
        if self.config.web_enabled:
            self.web_tools = WebTools(
                self.config.project_root, self.config.web_credentials_path,
                self.config.web_timeout_seconds)
            self.web_tools.check_configuration()
            self.tools.update(self.web_tools.registry)
        self.output_factory = output_factory if output_factory is not None else lambda: TerminalOutput(
            width=self.config.output_width, verbose=self.config.verbose)
        self._output = None
        self.conversation = Conversation(
            build_system_prompt(self.config.web_enabled),
            self.config.context_length, self.config.max_output_tokens)

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
        stream = None
        try:
            activity = getattr(output, "activity", None)
            if callable(activity):
                activity("Waiting for the model…")
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
            try:
                output.finish()
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
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
        output = self._get_output()
        begin_turn = getattr(output, "begin_turn", None)
        end_turn = getattr(output, "end_turn", None)
        compatibility_mode = False
        generated_tokens = 0
        counts_available = True
        completed = False
        try:
            if callable(begin_turn):
                begin_turn()
            while True:
                try:
                    response = self._request_response(compatibility_mode)
                except ResponseError as error:
                    if (compatibility_mode or error.status_code != 500
                            or "no user query found" not in error.error.lower()):
                        raise
                    output.status("Retrying the response…")
                    output.debug("[Chat] Using a compact conversation for Ollama compatibility.")
                    compatibility_mode = True
                    response = self._request_response(True)

                tokens = getattr(response, "eval_count", None)
                if type(tokens) is int and tokens >= 0:
                    generated_tokens += tokens
                else:
                    counts_available = False
                message = response.message
                self.conversation.messages.append(message)

                if not message.tool_calls:
                    reason = getattr(response, "done_reason", None) or "unknown"
                    token_label = f"{tokens:,}" if type(tokens) is int and tokens >= 0 else "unavailable"
                    output.debug(f"[Chat] Final model call ended: {reason}; generated tokens: {token_label}.")
                    if reason == "length":
                        output.warning("The response reached the generation limit and may be "
                                       "incomplete. Ask the model to continue, or narrow the request.")
                    elif not message.content:
                        output.warning("The model returned no answer and no tool calls. "
                                       "Try reformulating the question.")
                    completed = True
                    return

                for call in message.tool_calls:
                    self._execute_tool(call)
        finally:
            if callable(end_turn):
                end_turn(generated_tokens=generated_tokens if completed and counts_available else None)

    def _request_response(self, compatibility_mode):
        history = (self.conversation.compatibility_messages() if compatibility_mode
                   else self.conversation.messages)
        return self._stream_response(history)

    def _execute_tool(self, call):
        """Dispatch one tool call and save its evidence or error in the conversation."""
        name = call.function.name
        arguments = call.function.arguments
        output = self._get_output()
        try:
            output.tool(name, arguments)
            try:
                if name not in self.tools:
                    raise ValueError(f"Unknown tool: {name}")
                result = self.tools[name](**arguments)
            except Exception as error:
                result = json.dumps({"error": str(error)})
                output.error(f"{name}: {error}")
            self.conversation.messages.append({
                "role": "tool", "tool_name": name, "content": result,
            })
        finally:
            output.finish()

    def run(self):
        """Read terminal input and handle commands, failures, and interrupts."""
        output = self._get_output()
        startup = getattr(output, "startup", None)
        if callable(startup):
            startup(self.config.model, self.config.context_length,
                    self.pdf_tools.root, bool(self.web_tools))
        else:
            output.status(f"Model: {self.config.model}")
            output.status(f"Context window: {self.config.context_length:,} tokens")
            output.status(f"Project: {self.pdf_tools.root}")
            output.status(f"Web access: {'on (Tavily)' if self.web_tools else 'off'}")
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
