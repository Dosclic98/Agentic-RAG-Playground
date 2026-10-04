import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from ollama import ChatResponse, Message, ResponseError
from pypdf import PdfWriter
import pyte

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_agent as chat


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.agent = chat.PDFChatAgent(
            config=chat.AgentConfig(context_length=16384, max_output_tokens=8192), client=Mock())
        self.history = [{"role": "system", "content": chat.SYSTEM_PROMPT},
                        {"role": "user", "content": "Find Adobe annual reports."},
                        {"role": "assistant", "tool_calls": []},
                        {"role": "tool", "tool_name": "list_pdfs",
                         "content": '{"files": [{"path": "data/pdfs/ADOBE_2022_10K.pdf"}]}'}]
        self.agent.conversation.messages = self.history

    def test_specific_server_error_retries_with_user_and_evidence(self):
        message = Message(role="assistant", content="Found the report.")
        client = Mock()
        client.chat.side_effect = [
            ResponseError("no user query found in messages", 500),
            iter([ChatResponse(message=message, done=True)]),
        ]
        with patch.object(self.agent.conversation, "messages", self.history), patch.object(self.agent, "client", client), redirect_stdout(io.StringIO()):
            self.agent.answer()
        retry = client.chat.call_args.kwargs["messages"]
        self.assertEqual([item["role"] for item in retry], ["system", "user"])
        self.assertIn("Find Adobe annual reports.", retry[-1]["content"])
        self.assertIn("ADOBE_2022_10K.pdf", retry[-1]["content"])

    def test_other_server_errors_are_not_retried(self):
        client = Mock()
        client.chat.side_effect = ResponseError("model not found", 404)
        with patch.object(self.agent.conversation, "messages", self.history), patch.object(self.agent, "client", client):
            with self.assertRaises(ResponseError):
                self.agent.answer()
        self.assertEqual(client.chat.call_count, 1)

    def test_agents_have_independent_conversations_and_tool_registries(self):
        other = chat.PDFChatAgent(client=Mock())
        other.conversation.add_user("Keep this question.")
        self.agent.tools["custom_tool"] = Mock()
        with patch("builtins.input", side_effect=["/clear", "/exit"]), redirect_stdout(io.StringIO()):
            self.agent.run()
        self.assertEqual(len(self.agent.conversation.messages), 1)
        self.assertEqual(other.conversation.messages[1]["content"], "Keep this question.")
        self.assertNotIn("custom_tool", other.tools)

    def test_chat_dispatches_bound_pdf_tools_and_uses_instance_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "data/pdfs/report.pdf"
            pdf.parent.mkdir(parents=True)
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with pdf.open("wb") as output:
                writer.write(output)
            config = chat.AgentConfig(
                model="test-local-model", context_length=32768,
                max_output_tokens=1234, project_root=root)
            calls = [
                Message.ToolCall(function=Message.ToolCall.Function(
                    name="list_pdfs", arguments={})),
                Message.ToolCall(function=Message.ToolCall.Function(
                    name="read_pdf_content", arguments={"path": "data/pdfs/report.pdf"})),
            ]
            client = Mock()
            client.chat.side_effect = [
                iter([ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True)])
                for call in calls
            ] + [iter([ChatResponse(message=Message(
                role="assistant", content="The PDF has no extracted text."), done=True)])]
            agent = chat.PDFChatAgent(config=config, client=client)
            agent.conversation.add_user("Read report.pdf.")
            with redirect_stdout(io.StringIO()):
                agent.answer()
            evidence = [json.loads(message["content"])
                        for message in agent.conversation.messages
                        if isinstance(message, dict) and message["role"] == "tool"]
            self.assertEqual(evidence[0]["files"][0]["path"], "data/pdfs/report.pdf")
            self.assertEqual(evidence[1]["total_pages"], 1)
            self.assertFalse(evidence[1]["has_text"])
            self.assertEqual(client.chat.call_count, 3)
            for request in client.chat.call_args_list:
                self.assertEqual(request.kwargs["model"], config.model)
                self.assertEqual(request.kwargs["options"], {"num_ctx": 32768, "num_predict": 1234})

    def test_compact_history_preserves_question_when_evidence_overflows(self):
        self.history.append({"role": "tool", "content": "x" * 100000})
        compact = self.agent.conversation.compatibility_messages()
        self.assertIn("Find Adobe annual reports.", compact[-1]["content"])
        self.assertLess(len(compact[-1]["content"]), 25000)

    def test_failed_turn_remains_available_for_followup(self):
        history = [{"role": "system", "content": chat.SYSTEM_PROMPT}]
        with patch.object(self.agent.conversation, "messages", history), patch.object(self.agent, "answer", side_effect=RuntimeError("failed")), patch("builtins.input", side_effect=["Find Adobe annual reports.", "/exit"]), redirect_stdout(io.StringIO()):
            self.agent.run()
        self.assertEqual(history[1]["content"], "Find Adobe annual reports.")
        self.assertIn("failed", history[2]["content"])

    def test_token_limit_is_reported_and_partial_answer_is_kept(self):
        message = Message(role="assistant", content="Ho 8 annual report (2015-")
        response = ChatResponse(message=message, done=True, done_reason="length", eval_count=8192)
        output = io.StringIO()
        with patch.object(self.agent.conversation, "messages", self.history), patch.object(self.agent, "client", Mock(chat=Mock(return_value=iter([response])))), redirect_stdout(output):
            self.agent.answer()
        self.assertNotIn("Final model call ended: length", output.getvalue())
        self.assertIn("generation limit", output.getvalue())
        self.assertEqual(self.history[-1].content, message.content)

    def test_verbose_command_toggles_details_without_becoming_a_user_question(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/verbose", "/verbose off", "/verbose invalid", "/exit"]), redirect_stdout(output):
            self.agent.run()
        self.assertIn("Verbose output on", output.getvalue())
        self.assertIn("Verbose output off", output.getvalue())
        self.assertIn("Use /verbose", output.getvalue())
        self.assertEqual(len(self.history), 4)
        self.assertFalse(self.agent._get_output().verbose)

    def test_verbose_mode_reports_generation_statistics(self):
        client = Mock(chat=Mock(return_value=iter([ChatResponse(
            message=Message(role="assistant", content="Answer."), done=True,
            done_reason="stop", eval_count=20)])))
        agent = chat.PDFChatAgent(config=chat.AgentConfig(verbose=True), client=client)
        agent.conversation.add_user("Hello.")
        output = io.StringIO()
        with redirect_stdout(output):
            agent.answer()
        self.assertIn("Final model call ended: stop; generated tokens: 20", output.getvalue())

    def test_tool_failures_remain_visible_in_quiet_mode(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="unknown", arguments={}))
        output = io.StringIO()
        with redirect_stdout(output):
            self.agent._execute_tool(call)
        self.assertIn("Error: unknown: Unknown tool", output.getvalue())
        self.assertIn("error", self.history[-1]["content"])

    def test_large_search_results_keep_both_queries_and_sources(self):
        for query in ["operating revenues", "products brands"]:
            self.history.append({"role": "tool", "tool_name": "search_pdf",
                "content": json.dumps({"path": "data/pdfs/COCACOLA_2022_10K.pdf",
                    "query": query, "results": [{"page": 50 + i, "chunk_id": f"p{50+i}",
                        "text": query + " evidence " * 700} for i in range(8)]})})
        compact = self.agent.conversation.compatibility_messages()[-1]["content"]
        self.assertIn("operating revenues", compact)
        self.assertIn("products brands", compact)
        self.assertIn("COCACOLA_2022_10K.pdf", compact)
        self.assertIn('\\"page\\": 50', compact)
        self.assertIn("context_compacted", compact)
        self.assertLess(len(compact.encode("utf-8")), 25000)

    def test_compacted_content_cursor_does_not_skip_unseen_text(self):
        evidence = {"path": "data/pdfs/report.pdf", "page": 2, "offset": 100,
                    "text": "é" * 12000, "next": {"page": 3, "offset": 0}}
        compact = chat.Conversation._compact_record({"role": "tool", "tool_name": "read_pdf_content",
                                        "content": json.dumps(evidence)})
        content = json.loads(compact["content"])
        self.assertEqual(content["next"], {"page": 2, "offset": 100 + len(content["text"])})
        self.assertTrue(content["truncated"])

    def test_compacted_listing_cursor_keeps_omitted_files_accessible(self):
        evidence = {"offset": 200, "limit": 50, "has_more": False, "next_offset": None,
                    "files": [{"path": f"data/pdfs/report-{i}.pdf"} for i in range(50)]}
        record = chat.Conversation._compact_record({"role": "tool", "tool_name": "list_pdfs",
                                       "content": json.dumps(evidence)})
        shortened = json.loads(record["content"])
        self.assertEqual(shortened["next_offset"], 200 + len(shortened["files"]))
        self.assertTrue(shortened["has_more"])

    def test_compaction_does_not_modify_saved_evidence(self):
        record = {"role": "tool", "content": json.dumps({"text": "x" * 100000})}
        original = record.copy()
        chat.Conversation._compact_record(record)
        self.assertEqual(record, original)

    def test_web_disabled_sends_no_web_schemas_or_instructions_and_reads_no_credentials(self):
        with patch.object(chat.WebTools, "check_configuration", side_effect=AssertionError("unused")):
            agent = chat.PDFChatAgent(client=Mock())
        self.assertIsNone(agent.web_tools)
        self.assertEqual(agent.conversation.messages[0]["content"], chat.SYSTEM_PROMPT)
        self.assertNotIn("web_search", agent.tools)
        client = Mock(chat=Mock(return_value=iter([ChatResponse(
            message=Message(role="assistant", content="Hello."), done=True)])))
        agent.client = client
        with redirect_stdout(io.StringIO()):
            agent._stream_response([{"role": "user", "content": "Hello"}])
        self.assertNotIn("web_search", [tool.__name__ for tool in client.chat.call_args.kwargs["tools"]])

    def test_enabled_web_tools_search_read_and_answer_with_saved_url_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".web_credentials.json").write_text('{"tavily_api_key": "test-key"}')
            config = chat.AgentConfig(project_root=root, web_enabled=True)
            url = "https://example.com/announcement"
            calls = [Message.ToolCall(function=Message.ToolCall.Function(
                name="web_search", arguments={"query": "public announcement"})),
                Message.ToolCall(function=Message.ToolCall.Function(name="read_web_page", arguments={"url": url}))]
            content = f"External context: the announcement is published. [{url}]"
            client = Mock(chat=Mock(side_effect=[
                iter([ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True)])
                for call in calls] + [iter([ChatResponse(message=Message(role="assistant", content=content), done=True)])]))
            agent = chat.PDFChatAgent(config=config, client=client)
            agent.conversation.add_user("Find the latest public announcement.")
            with patch.object(agent.web_tools, "_post", side_effect=[
                {"results": [{"title": "Announcement", "url": url, "content": "A useful lead."}]},
                {"results": [{"url": url, "raw_content": "The official announcement."}]}]), redirect_stdout(io.StringIO()):
                agent.answer()
            evidence = [json.loads(message["content"]) for message in agent.conversation.messages
                        if isinstance(message, dict) and message["role"] == "tool"]
            self.assertEqual(evidence[0]["source_type"], "web")
            self.assertEqual(evidence[1]["text"], "The official announcement.")
            self.assertEqual(agent.conversation.messages[-1].content, content)
            self.assertIn("Optional web retrieval is enabled", agent.conversation.messages[0]["content"])
            self.assertEqual(len(agent.tools), len(agent.pdf_tools.registry) + 2)
            from ollama._utils import convert_function_to_tool
            schemas = [convert_function_to_tool(tool).model_dump() for tool in agent.web_tools.registry.values()]
            self.assertEqual(schemas[0]["function"]["parameters"]["required"], ["query"])
            self.assertEqual(schemas[1]["function"]["parameters"]["required"], ["url"])
            self.assertNotIn("test-key", json.dumps(schemas))

    def test_enabled_web_without_credentials_fails_before_model_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            with self.assertRaisesRegex(ValueError, "tavily_api_key"):
                chat.PDFChatAgent(config=chat.AgentConfig(project_root=Path(directory), web_enabled=True), client=client)
            client.chat.assert_not_called()

    def test_compacted_web_page_preserves_url_and_resumes_after_visible_characters(self):
        url = "https://example.com/report?query=" + "x" * 1500
        evidence = {"source_type": "web", "url": url, "requested_url": url,
                    "offset": 100, "text": "é" * 12000, "next": None}
        record = {"role": "tool", "tool_name": "read_web_page", "content": json.dumps(evidence)}
        shortened = json.loads(chat.Conversation._compact_record(record)["content"])
        self.assertEqual(shortened["url"], url)
        self.assertEqual(shortened["requested_url"], url)
        self.assertEqual(shortened["next"], {"url": url, "offset": 100 + len(shortened["text"])})
        self.assertTrue(shortened["truncated"])
        self.assertEqual(json.loads(record["content"]), evidence)

    def test_compacted_collection_search_cursor_keeps_omitted_passages_accessible(self):
        evidence = {"offset": 10, "next_offset": None, "has_more": False,
                    "results": [{"path": f"data/pdfs/report-{i}.pdf", "page": 1,
                                 "text": "evidence"} for i in range(10)]}
        record = chat.Conversation._compact_record({"role": "tool", "tool_name": "search_collection",
                                                    "content": json.dumps(evidence)})
        shortened = json.loads(record["content"])
        self.assertEqual(shortened["next_offset"], 10 + len(shortened["results"]))
        self.assertTrue(shortened["has_more"])

    def test_compacted_table_cursor_does_not_skip_omitted_rows(self):
        evidence = {"path": "data/pdfs/report.pdf", "page": 2, "table_index": 1, "row_offset": 20,
                    "table": {"rows": [[str(index), "value"] for index in range(50)]},
                    "next": {"page": 2, "table_index": 2, "row_offset": 0}}
        record = chat.Conversation._compact_record({"role": "tool", "tool_name": "extract_pdf_tables",
                                                    "content": json.dumps(evidence)})
        shortened = json.loads(record["content"])
        self.assertEqual(shortened["next"], {"page": 2, "table_index": 1,
                                            "row_offset": 20 + len(shortened["table"]["rows"])})
        self.assertTrue(shortened["truncated"])

    def test_compacted_ocr_cursor_resumes_after_visible_text(self):
        evidence = {"path": "data/pdfs/report.pdf", "pages": [
            {"page": 2, "offset": 0, "text": "é" * 12000,
             "next": {"page": 2, "offset": 12000}}]}
        record = chat.Conversation._compact_record({"role": "tool", "tool_name": "ocr_pdf_pages",
                                                    "content": json.dumps(evidence)})
        page = json.loads(record["content"])["pages"][0]
        self.assertEqual(page["next"], {"page": 2, "offset": len(page["text"])})

    def test_larger_context_keeps_more_evidence_and_is_sent_to_ollama(self):
        for index in range(12):
            self.history.append({"role": "tool", "tool_name": "search_pdf",
                "content": json.dumps({"query": f"evidence-{index}", "text": "x" * 4000})})
        small = self.agent.conversation.compatibility_messages()[-1]["content"]
        message = Message(role="assistant", content="Answer.")
        client = Mock(chat=Mock(return_value=iter([ChatResponse(message=message, done=True)])))
        large_agent = chat.PDFChatAgent(
            config=chat.AgentConfig(context_length=65536), client=client)
        large_agent.conversation.messages = self.history
        large = large_agent.conversation.compatibility_messages()[-1]["content"]
        with redirect_stdout(io.StringIO()):
            large_agent.answer()
        self.assertNotIn("evidence-0", small)
        self.assertIn("evidence-0", large)
        self.assertEqual(client.chat.call_args.kwargs["options"]["num_ctx"], 65536)

    def test_stream_accumulates_thinking_answer_and_tool_calls(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="list_pdfs", arguments={"path": "data"}))
        chunks = [
            ChatResponse(message=Message(role="assistant", thinking="First thought.\n")),
            ChatResponse(message=Message(role="assistant", thinking="Second thought.")),
            ChatResponse(message=Message(role="assistant", content="Found ")),
            ChatResponse(message=Message(role="assistant", content="a report.", tool_calls=[call])),
            ChatResponse(message=Message(role="assistant"), done=True, done_reason="stop", eval_count=30),
        ]
        output = io.StringIO()
        client = Mock(chat=Mock(return_value=iter(chunks)))
        with patch.object(self.agent, "client", client), redirect_stdout(output):
            response = self.agent._stream_response(self.history)
        self.assertEqual(response.message.content, "Found a report.")
        self.assertEqual(response.message.thinking, "First thought.\nSecond thought.")
        self.assertEqual(response.message.tool_calls, [call])
        self.assertEqual(response.eval_count, 30)
        self.assertIn("Assistant: Found a report.", output.getvalue())
        self.assertNotIn("First thought", output.getvalue())
        self.assertNotIn("\033", output.getvalue())
        self.assertTrue(client.chat.call_args.kwargs["stream"])
        self.assertTrue(client.chat.call_args.kwargs["think"])

    def test_interrupted_stream_clears_thinking(self):
        def chunks():
            yield ChatResponse(message=Message(role="assistant", thinking="Partial thought"))
            raise KeyboardInterrupt
        client = Mock(chat=Mock(return_value=chunks()))
        display = Mock()
        with patch.object(self.agent, "client", client), patch.object(self.agent, "output_factory", return_value=display):
            with self.assertRaises(KeyboardInterrupt):
                self.agent._stream_response(self.history)
        display.finish.assert_called_once()

    def test_stream_error_can_trigger_compatibility_retry(self):
        def broken_stream():
            raise ResponseError("no user query found in messages", 500)
            yield
        client = Mock()
        client.chat.side_effect = [broken_stream(), iter([
            ChatResponse(message=Message(role="assistant", content="Recovered."), done=True)])]
        with patch.object(self.agent, "client", client), patch.object(self.agent.conversation, "messages", self.history), redirect_stdout(io.StringIO()):
            self.agent.answer()
        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(self.history[-1].content, "Recovered.")

    def test_tool_loop_continues_beyond_eight_rounds(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        replies = [ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True)
                   for _ in range(10)]
        replies.append(ChatResponse(message=Message(role="assistant", content="Complete."), done=True))
        lookup = Mock(return_value='{"text": "evidence"}')
        with patch.object(self.agent.conversation, "messages", self.history), patch.object(self.agent, "tools", {"lookup": lookup}), patch.object(self.agent, "_stream_response", side_effect=replies) as request, redirect_stdout(io.StringIO()):
            self.agent.answer()
        self.assertEqual(lookup.call_count, 10)
        self.assertEqual(request.call_count, 11)
        self.assertEqual(self.history[-1].content, "Complete.")

    def test_tool_loop_remains_interruptible(self):
        with patch.object(self.agent, "_stream_response", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self.agent.answer()

    @staticmethod
    def renderer_mock(include_hooks=True):
        methods = ["status", "debug", "warning", "error", "tool", "thinking",
                   "answer", "clear_thinking", "finish", "prompt", "toggle_verbose"]
        if include_hooks:
            methods += ["startup", "begin_turn", "activity", "end_turn"]
        return Mock(spec=methods)

    def test_startup_hook_receives_settings_without_duplicate_status_lines(self):
        output = self.renderer_mock()
        output.prompt.return_value = "/exit"
        with patch.object(self.agent, "output_factory", return_value=output):
            self.agent.run()
        output.startup.assert_called_once_with(
            self.agent.config.model, self.agent.config.context_length,
            self.agent.pdf_tools.root, False)
        output.status.assert_not_called()

    def test_legacy_renderer_keeps_startup_and_answer_without_optional_hooks(self):
        output = self.renderer_mock(include_hooks=False)
        output.prompt.return_value = "/exit"
        response = ChatResponse(message=Message(role="assistant", content="Answer."),
                                done=True, eval_count=10)
        client = Mock(chat=Mock(return_value=iter([response])))
        with patch.object(self.agent, "output_factory", return_value=output), patch.object(self.agent, "client", client):
            self.agent.run()
            self.agent.answer()
        self.assertEqual(output.status.call_count, 5)
        self.assertIn("Context window: 16,384", output.status.call_args_list[1].args[0])
        output.answer.assert_called_once_with("Answer.")
        output.finish.assert_called_once()

    def test_turn_hooks_span_all_tool_rounds_and_sum_actual_generation_counts(self):
        output = self.renderer_mock()
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        replies = [ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True, eval_count=12),
                   ChatResponse(message=Message(role="assistant", content="Answer."), done=True, eval_count=30)]
        client = Mock(chat=Mock(side_effect=[iter([reply]) for reply in replies]))
        with patch.object(self.agent, "output_factory", return_value=output), patch.object(self.agent, "client", client), patch.object(self.agent, "tools", {"lookup": Mock(return_value='{"text": "evidence"}')}):
            self.agent.answer()
        output.begin_turn.assert_called_once_with()
        output.end_turn.assert_called_once_with(generated_tokens=42)
        output.debug.assert_called_once_with(
            "[Chat] Final model call ended: unknown; generated tokens: 30.")
        self.assertEqual([entry.args[0] for entry in output.activity.call_args_list],
                         ["Waiting for the model…", "Waiting for the model…"])
        self.assertEqual(output.finish.call_count, 3)
        self.assertEqual(output.method_calls[0][0], "begin_turn")
        self.assertEqual(output.method_calls[-1][0], "end_turn")

    def test_turn_footer_omits_unknown_counts_and_preserves_measured_zero(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        for first_count, final_count, expected in [(None, 30, None), (12, None, None), (0, 0, 0)]:
            with self.subTest(first=first_count, final=final_count):
                output = self.renderer_mock()
                replies = [ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True, eval_count=first_count),
                           ChatResponse(message=Message(role="assistant", content="Answer."), done=True, eval_count=final_count)]
                client = Mock(chat=Mock(side_effect=[iter([reply]) for reply in replies]))
                agent = chat.PDFChatAgent(client=client, output_factory=lambda: output)
                agent.tools = {"lookup": Mock(return_value='{"text": "evidence"}')}
                agent.conversation.add_user("Find evidence.")
                agent.answer()
                output.end_turn.assert_called_once_with(generated_tokens=expected)

    def test_initial_client_failure_finishes_display_and_turn(self):
        output = self.renderer_mock()
        client = Mock(chat=Mock(side_effect=RuntimeError("offline")))
        with patch.object(self.agent, "output_factory", return_value=output), patch.object(self.agent, "client", client):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                self.agent.answer()
        output.activity.assert_called_once_with("Waiting for the model…")
        output.finish.assert_called_once_with()
        output.begin_turn.assert_called_once_with()
        output.end_turn.assert_called_once_with(generated_tokens=None)

    def test_interrupted_later_stream_closes_resources_and_turn_without_partial_totals(self):
        output = self.renderer_mock()
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        first = ChatResponse(message=Message(role="assistant", tool_calls=[call]), done=True, eval_count=12)

        class InterruptedStream:
            close = Mock()

            def __iter__(self):
                yield ChatResponse(message=Message(role="assistant", thinking="Partial thought"))
                raise KeyboardInterrupt

        stream = InterruptedStream()
        client = Mock(chat=Mock(side_effect=[iter([first]), stream]))
        with patch.object(self.agent, "output_factory", return_value=output), patch.object(self.agent, "client", client), patch.object(self.agent, "tools", {"lookup": Mock(return_value='{"text": "evidence"}')}):
            with self.assertRaises(KeyboardInterrupt):
                self.agent.answer()
        stream.close.assert_called_once_with()
        self.assertEqual(output.finish.call_count, 3)
        output.begin_turn.assert_called_once_with()
        output.end_turn.assert_called_once_with(generated_tokens=None)

    def test_tool_completion_errors_and_interrupts_always_finish_activity(self):
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        evidence = '{"text": "evidence"}'
        for exception in (None, ValueError("tool failed"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                output = self.renderer_mock()
                agent = chat.PDFChatAgent(client=Mock(), output_factory=lambda: output)
                agent.tools = {"lookup": Mock(return_value=evidence, side_effect=exception)}
                if isinstance(exception, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        agent._execute_tool(call)
                    self.assertEqual(len(agent.conversation.messages), 1)
                else:
                    agent._execute_tool(call)
                    content = agent.conversation.messages[-1]["content"]
                    if exception is None:
                        self.assertEqual(content, evidence)
                    else:
                        self.assertIn("tool failed", content)
                output.finish.assert_called_once_with()
                output.activity.assert_not_called()

    def test_standalone_interrupted_tool_stops_live_renderer_and_restores_cursor(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        stream = Terminal()
        call = Message.ToolCall(function=Message.ToolCall.Function(name="lookup", arguments={}))
        with patch.dict(os.environ, {"TERM": "xterm-256color", "TTY_INTERACTIVE": "1", "NO_COLOR": ""}), patch("terminal_output.shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            output = chat.TerminalOutput(stream)
            agent = chat.PDFChatAgent(client=Mock(), output_factory=lambda: output)
            agent.tools = {"lookup": Mock(side_effect=KeyboardInterrupt)}
            with self.assertRaises(KeyboardInterrupt):
                agent._execute_tool(call)
        self.assertIsNone(output._live)
        self.assertIsNone(output._activity)
        screen = pyte.Screen(80, 24)
        pyte.Stream(screen).feed(stream.getvalue().replace("\n", "\r\n"))
        self.assertFalse(screen.cursor.hidden)

    def test_stream_closes_even_when_display_finish_fails(self):
        output = self.renderer_mock()
        output.finish.side_effect = RuntimeError("display failed")

        class CompletedStream:
            close = Mock()

            def __iter__(self):
                yield ChatResponse(message=Message(role="assistant", content="Answer."), done=True)

        stream = CompletedStream()
        client = Mock(chat=Mock(return_value=stream))
        with patch.object(self.agent, "output_factory", return_value=output), patch.object(self.agent, "client", client):
            with self.assertRaisesRegex(RuntimeError, "display failed"):
                self.agent._stream_response(self.history)
        stream.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
