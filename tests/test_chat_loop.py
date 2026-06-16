from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from fizzy import change_history
from fizzy.chat_loop import ChatLoop
from fizzy.edit_applier import EditOutcome
from fizzy.prompts import AGENT_INSTRUCTIONS
from fizzy.session import FileEntry, Session
from fizzy.token_tracker import TokenStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(inputs: list, stream_chunks: list[str] = None, token_status=TokenStatus.OK):
    """Build a ChatLoop with fully mocked dependencies.

    `inputs` is the sequence of strings returned by read_line (EOFError signals end).
    `stream_chunks` is what the LLM client yields per call.
    """
    session = Session(
        model="claude-3-5-sonnet-20241022",
        working_dir=Path("/tmp"),
        max_tokens=10_000,
    )

    client = MagicMock()
    client.stream.return_value = iter(stream_chunks or ["Hello there!"])

    reader = MagicMock()
    # Convert the input list into sequential return values; last entry is EOFError
    side_effects = list(inputs) + [EOFError]
    reader.read_line.side_effect = side_effects

    renderer = MagicMock()
    # start_stream returns a context manager
    live_ctx = MagicMock()
    live_ctx.__enter__ = MagicMock(return_value=live_ctx)
    live_ctx.__exit__ = MagicMock(return_value=False)
    renderer.start_stream.return_value = live_ctx

    tracker = MagicMock()
    tracker.check.return_value = (token_status, 500)

    loop = ChatLoop(
        session=session,
        client=client,
        reader=reader,
        renderer=renderer,
        tracker=tracker,
    )
    return loop, session, client, reader, renderer, tracker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalFlow:
    def test_user_message_added_to_history(self):
        loop, session, *_ = _make_loop(["hi"])
        loop.run()
        assert any(m["content"] == "hi" for m in session.history)

    def test_assistant_reply_added_to_history(self):
        loop, session, *_ = _make_loop(["hi"], stream_chunks=["Hello", "!"])
        loop.run()
        assert any(m["role"] == "assistant" and "Hello!" in m["content"] for m in session.history)

    def test_stream_chunks_are_accumulated(self):
        loop, session, client, _, renderer, _ = _make_loop(
            ["hi"], stream_chunks=["chunk1", " chunk2"]
        )
        loop.run()
        # finish_stream should be called with the full accumulated text
        renderer.finish_stream.assert_called_once()
        _, accumulated = renderer.finish_stream.call_args.args
        assert accumulated == "chunk1 chunk2"

    def test_token_status_rendered_after_each_turn(self):
        loop, _, _, _, _, tracker = _make_loop(["hello", "world"])
        loop.run()
        assert tracker.render_status.call_count == 2

    def test_multiple_turns_all_recorded(self):
        loop, session, *_ = _make_loop(
            ["first", "second"], stream_chunks=["ok"]
        )
        # Need fresh stream per call
        loop._client.stream.side_effect = [iter(["reply1"]), iter(["reply2"])]
        loop.run()
        roles = [m["role"] for m in session.history]
        assert roles == ["user", "assistant", "user", "assistant"]


class TestEmptyInput:
    def test_empty_string_is_skipped(self):
        loop, session, client, *_ = _make_loop(["", "hello"])
        loop.run()
        # LLM called only once (for "hello", not for "")
        assert client.stream.call_count == 1

    def test_whitespace_only_skipped(self):
        """read_line strips input; stripped whitespace becomes empty string."""
        loop, session, client, reader, *_ = _make_loop(["hello"])
        # Inject whitespace as the first call
        reader.read_line.side_effect = ["   ", "hello", EOFError]
        loop.run()
        assert client.stream.call_count == 1


class TestExitCommands:
    @pytest.mark.parametrize("cmd", ["/quit", "/exit", "/q", "/QUIT", "/EXIT"])
    def test_exit_command_stops_loop(self, cmd):
        loop, _, client, *_ = _make_loop([cmd])
        loop.run()
        client.stream.assert_not_called()

    def test_eof_stops_loop(self):
        loop, _, client, reader, renderer, _ = _make_loop([])
        # read_line immediately raises EOFError
        reader.read_line.side_effect = [EOFError]
        loop.run()
        client.stream.assert_not_called()
        renderer.print_info.assert_called()  # goodbye message


class TestTokenBlocking:
    def test_blocked_message_not_sent(self):
        loop, session, client, *_ = _make_loop(["hi"], token_status=TokenStatus.BLOCK)
        loop.run()
        client.stream.assert_not_called()

    def test_blocked_message_removed_from_history(self):
        loop, session, *_ = _make_loop(["hi"], token_status=TokenStatus.BLOCK)
        loop.run()
        assert session.history == []

    def test_blocked_prints_error(self):
        loop, _, _, _, renderer, _ = _make_loop(["hi"], token_status=TokenStatus.BLOCK)
        loop.run()
        renderer.print_error.assert_called_once()

    def test_warn_still_sends(self):
        loop, _, client, *_ = _make_loop(["hi"], token_status=TokenStatus.WARN)
        loop.run()
        client.stream.assert_called_once()

    def test_warn_prints_info(self):
        loop, _, _, _, renderer, _ = _make_loop(["hi"], token_status=TokenStatus.WARN)
        loop.run()
        # print_info used for warn message and for the startup banner — at least 2 calls
        assert renderer.print_info.call_count >= 2


class TestStreamErrors:
    def test_runtime_error_from_client_prints_error(self):
        loop, session, client, _, renderer, _ = _make_loop(["hi"])
        client.stream.side_effect = RuntimeError("Authentication failed — check your API key.")
        loop.run()
        renderer.print_error.assert_called()
        error_msg = renderer.print_error.call_args.args[0]
        assert "Authentication failed" in error_msg

    def test_runtime_error_rolls_back_user_message(self):
        loop, session, client, *_ = _make_loop(["hi"])
        client.stream.side_effect = RuntimeError("auth error")
        loop.run()
        assert session.history == []

    def test_unexpected_exception_prints_error(self):
        loop, session, client, _, renderer, _ = _make_loop(["hi"])
        client.stream.side_effect = ValueError("something unexpected")
        loop.run()
        renderer.print_error.assert_called()

    def test_unexpected_exception_rolls_back_message(self):
        loop, session, client, *_ = _make_loop(["hi"])
        client.stream.side_effect = ValueError("boom")
        loop.run()
        assert session.history == []

    def test_loop_continues_after_error(self):
        loop, session, client, reader, *_ = _make_loop(["first", "second"])
        # First call fails, second succeeds
        client.stream.side_effect = [RuntimeError("fail"), iter(["ok"])]
        loop.run()
        # "second" message should be in history with its reply
        assert any(m["content"] == "second" for m in session.history)
        assert any(m["content"] == "ok" for m in session.history)


# ---------------------------------------------------------------------------
# /add command dispatch
# ---------------------------------------------------------------------------

class TestAddCommand:
    def test_add_does_not_call_stream(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(True, "Added 'foo.py'.")):
            loop, _, client, *_ = _make_loop(["/add foo.py"])
            loop.run()
            client.stream.assert_not_called()

    def test_add_success_prints_info_with_filename(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(True, "Added 'foo.py'.")):
            loop, _, _, _, renderer, _ = _make_loop(["/add foo.py"])
            loop.run()
            assert any("foo.py" in str(c) for c in renderer.print_info.call_args_list)

    def test_add_failure_prints_error(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(False, "Error: 'ghost.py' not found.")):
            loop, _, _, _, renderer, _ = _make_loop(["/add ghost.py"])
            loop.run()
            renderer.print_error.assert_called()
            assert "ghost.py" in str(renderer.print_error.call_args)

    def test_add_failure_does_not_call_stream(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(False, "Error: not found.")):
            loop, _, client, *_ = _make_loop(["/add ghost.py"])
            loop.run()
            client.stream.assert_not_called()

    def test_add_no_filename_prints_usage_hint(self):
        loop, _, client, _, renderer, _ = _make_loop(["/add"])
        loop.run()
        client.stream.assert_not_called()
        assert any("/add" in str(c) for c in renderer.print_info.call_args_list)

    def test_add_whitespace_in_arg_prints_usage_hint(self):
        loop, _, client, _, renderer, _ = _make_loop(["/add foo bar"])
        loop.run()
        client.stream.assert_not_called()
        assert any("/add" in str(c) for c in renderer.print_info.call_args_list)

    def test_add_command_not_stored_in_history(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(True, "Added.")):
            loop, session, *_ = _make_loop(["/add foo.py"])
            loop.run()
            assert session.history == []

    def test_normal_message_after_add_still_sent_to_llm(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(True, "Added.")):
            loop, _, client, *_ = _make_loop(["/add foo.py", "hello"])
            loop.run()
            client.stream.assert_called_once()


# ---------------------------------------------------------------------------
# System message injection
# ---------------------------------------------------------------------------
# refresh_files is patched to [] in this class so pre-seeded context_files
# entries (pointing to non-existent paths) are not evicted before the LLM call.

class TestSystemMessageInjection:
    def test_no_files_stream_receives_no_system_message(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, _, client, *_ = _make_loop(["hello"])
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert all(m["role"] != "system" for m in messages_sent)

    def test_with_files_stream_first_message_is_system(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert messages_sent[0]["role"] == "system"

    def test_system_message_content_contains_file_body(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(
                content="UNIQUE_MARKER = True\n", mtime=1.0
            )
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            # messages[0] = agent instructions; messages[1] = file contents
            assert "UNIQUE_MARKER = True" in messages_sent[1]["content"]

    def test_system_message_not_stored_in_session_history(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            assert all(m["role"] != "system" for m in session.history)

    def test_user_message_still_present_after_system_message(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert any(m["role"] == "user" and m["content"] == "hello" for m in messages_sent)

    def test_with_files_first_system_message_is_agent_instructions(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert messages_sent[0]["role"] == "system"
            assert messages_sent[0]["content"] == AGENT_INSTRUCTIONS

    def test_with_files_second_system_message_is_file_contents(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(
                content="UNIQUE_MARKER = True\n", mtime=1.0
            )
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert messages_sent[1]["role"] == "system"
            assert "UNIQUE_MARKER = True" in messages_sent[1]["content"]

    def test_with_files_exactly_two_system_messages_prepended(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, client, *_ = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            system_count = sum(1 for m in messages_sent if m["role"] == "system")
            assert system_count == 2

    def test_no_files_agent_instructions_not_injected(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, _, client, *_ = _make_loop(["hello"])
            loop.run()
            messages_sent = client.stream.call_args.args[0]
            assert not any(
                m["role"] == "system" and m.get("content") == AGENT_INSTRUCTIONS
                for m in messages_sent
            )


# ---------------------------------------------------------------------------
# refresh_files wiring
# ---------------------------------------------------------------------------

class TestRefreshOnLLMTurn:
    def test_refresh_called_once_per_llm_turn(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]) as mock_refresh:
            loop, *_ = _make_loop(["hello", "world"])
            loop._client.stream.side_effect = [iter(["r1"]), iter(["r2"])]
            loop.run()
            assert mock_refresh.call_count == 2

    def test_refresh_not_called_for_add_command(self):
        with patch("fizzy.chat_loop.file_context.add_file", return_value=(True, "Added.")):
            with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]) as mock_refresh:
                loop, *_ = _make_loop(["/add foo.py"])
                loop.run()
                mock_refresh.assert_not_called()

    def test_refresh_notices_printed_before_llm_response(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=["'foo.py' reloaded."]):
            loop, _, _, _, renderer, _ = _make_loop(["hello"])
            loop.run()
            assert any("foo.py" in str(c) for c in renderer.print_info.call_args_list)

    def test_multiple_refresh_notices_all_printed(self):
        notices = ["'a.py' reloaded.", "'b.py' removed from context (deleted on disk)."]
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=notices):
            loop, _, _, _, renderer, _ = _make_loop(["hello"])
            loop.run()
            all_info = " ".join(str(c) for c in renderer.print_info.call_args_list)
            assert "a.py" in all_info
            assert "b.py" in all_info


# ---------------------------------------------------------------------------
# Token counting includes the system message
# ---------------------------------------------------------------------------

class TestTokenCountingWithSystemMessage:
    def test_tracker_receives_system_message_when_files_in_context(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, _, _, _, tracker = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            # First tracker.check call (pre-send budget check)
            first_call_messages = tracker.check.call_args_list[0].args[0]
            assert first_call_messages[0]["role"] == "system"

    def test_tracker_receives_only_history_when_no_files(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, _, _, _, _, tracker = _make_loop(["hello"])
            loop.run()
            first_call_messages = tracker.check.call_args_list[0].args[0]
            assert all(m["role"] != "system" for m in first_call_messages)

    def test_tracker_post_send_check_also_includes_system_message(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, _, _, _, tracker = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            # Second tracker.check call (post-send render_status check)
            second_call_messages = tracker.check.call_args_list[1].args[0]
            assert second_call_messages[0]["role"] == "system"

    def test_tracker_pre_send_first_message_is_agent_instructions(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, _, _, _, tracker = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            first_call_messages = tracker.check.call_args_list[0].args[0]
            assert first_call_messages[0]["content"] == AGENT_INSTRUCTIONS

    def test_tracker_post_send_first_message_is_agent_instructions(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, session, _, _, _, tracker = _make_loop(["hello"])
            session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
            loop.run()
            second_call_messages = tracker.check.call_args_list[1].args[0]
            assert second_call_messages[0]["content"] == AGENT_INSTRUCTIONS


# ---------------------------------------------------------------------------
# /drop command dispatch
# ---------------------------------------------------------------------------

class TestDropCommand:
    def test_drop_does_not_call_stream(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(True, "Dropped 'foo.py'.")):
            loop, _, client, *_ = _make_loop(["/drop foo.py"])
            loop.run()
            client.stream.assert_not_called()

    def test_drop_success_prints_info_with_filename(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(True, "Dropped 'foo.py'.")):
            loop, _, _, _, renderer, _ = _make_loop(["/drop foo.py"])
            loop.run()
            assert any("foo.py" in str(c) for c in renderer.print_info.call_args_list)

    def test_drop_failure_prints_error_with_filename(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(False, "Error: 'ghost.py' is not in context.")):
            loop, _, _, _, renderer, _ = _make_loop(["/drop ghost.py"])
            loop.run()
            renderer.print_error.assert_called()
            assert "ghost.py" in str(renderer.print_error.call_args)

    def test_drop_failure_does_not_call_stream(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(False, "Error: not in context.")):
            loop, _, client, *_ = _make_loop(["/drop ghost.py"])
            loop.run()
            client.stream.assert_not_called()

    def test_drop_no_filename_prints_usage_hint(self):
        loop, _, client, _, renderer, _ = _make_loop(["/drop"])
        loop.run()
        client.stream.assert_not_called()
        assert any("/drop" in str(c) for c in renderer.print_info.call_args_list)

    def test_drop_whitespace_in_arg_prints_usage_hint(self):
        loop, _, client, _, renderer, _ = _make_loop(["/drop foo bar"])
        loop.run()
        client.stream.assert_not_called()
        assert any("/drop" in str(c) for c in renderer.print_info.call_args_list)

    def test_drop_command_not_stored_in_history(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(True, "Dropped.")):
            loop, session, *_ = _make_loop(["/drop foo.py"])
            loop.run()
            assert session.history == []

    def test_normal_message_after_drop_still_sent_to_llm(self):
        with patch("fizzy.chat_loop.file_context.drop_file", return_value=(True, "Dropped.")):
            loop, _, client, *_ = _make_loop(["/drop foo.py", "hello"])
            loop.run()
            client.stream.assert_called_once()


# ---------------------------------------------------------------------------
# /files command dispatch
# ---------------------------------------------------------------------------

class TestFilesCommand:
    def test_files_does_not_call_stream(self):
        loop, _, client, *_ = _make_loop(["/files"])
        loop.run()
        client.stream.assert_not_called()

    def test_files_empty_context_prints_info(self):
        loop, _, _, _, renderer, _ = _make_loop(["/files"])
        loop.run()
        renderer.print_info.assert_called()

    def test_files_empty_context_message_says_no_files(self):
        loop, _, _, _, renderer, _ = _make_loop(["/files"])
        loop.run()
        all_info = " ".join(str(c) for c in renderer.print_info.call_args_list).lower()
        assert "no files" in all_info

    def test_files_shows_relative_path(self):
        loop, session, _, _, renderer, _ = _make_loop(["/files"])
        # working_dir is Path("/tmp"); relative display should show just "foo.py"
        session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
        loop.run()
        all_info = " ".join(str(c) for c in renderer.print_info.call_args_list)
        assert "foo.py" in all_info

    def test_files_does_not_show_absolute_path_prefix(self):
        loop, session, _, _, renderer, _ = _make_loop(["/files"])
        session.context_files[Path("/tmp/foo.py")] = FileEntry(content="x = 1\n", mtime=1.0)
        loop.run()
        all_info = " ".join(str(c) for c in renderer.print_info.call_args_list)
        # relative display: "foo.py" present, full absolute "/tmp/foo.py" absent
        assert "/tmp/foo.py" not in all_info

    def test_files_multiple_files_all_shown(self):
        loop, session, _, _, renderer, _ = _make_loop(["/files"])
        session.context_files[Path("/tmp/alpha.py")] = FileEntry(content="a\n", mtime=1.0)
        session.context_files[Path("/tmp/beta.py")] = FileEntry(content="b\n", mtime=1.0)
        loop.run()
        all_info = " ".join(str(c) for c in renderer.print_info.call_args_list)
        assert "alpha.py" in all_info
        assert "beta.py" in all_info

    def test_files_command_not_stored_in_history(self):
        loop, session, *_ = _make_loop(["/files"])
        loop.run()
        assert session.history == []

    def test_normal_message_after_files_still_sent_to_llm(self):
        loop, _, client, *_ = _make_loop(["/files", "hello"])
        loop.run()
        client.stream.assert_called_once()


# ---------------------------------------------------------------------------
# Edit applier integration
# ---------------------------------------------------------------------------

class TestEditApplierIntegration:
    def test_parse_proposals_called_with_assistant_reply(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]) as mock_parse:
            loop, session, *_ = _make_loop(["hello"], stream_chunks=["the reply"])
            loop.run()
            mock_parse.assert_called_once()
            args = mock_parse.call_args.args
            assert args[0] == "the reply"   # full accumulated reply
            assert args[1] is session        # same session object

    def test_apply_edits_called_with_all_proposals(self):
        # The whole proposal list is handed to apply_edits, which groups by file.
        p1, p2 = MagicMock(), MagicMock()
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[p1, p2]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits", return_value=[]) as mock_apply:
                loop, session, *_ = _make_loop(["hello"])
                loop.run()
                mock_apply.assert_called_once()
                assert mock_apply.call_args.args[0] == [p1, p2]
                assert mock_apply.call_args.args[1] is session

    def test_apply_edits_not_called_when_no_proposals(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits") as mock_apply:
                loop, *_ = _make_loop(["hello"])
                loop.run()
                mock_apply.assert_not_called()

    def test_proposals_processed_after_assistant_message_stored(self):
        """Proposals are processed after add_assistant_message, so history is complete."""
        call_order = []

        def track_parse(response, session):
            call_order.append(("parse", len(session.history)))
            return []

        with patch("fizzy.chat_loop.edit_applier.parse_proposals", side_effect=track_parse):
            loop, *_ = _make_loop(["hello"], stream_chunks=["reply"])
            loop.run()

        # history should have both user + assistant by the time parse is called
        assert call_order == [("parse", 2)]

    def test_edit_applier_not_invoked_for_slash_commands(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals") as mock_parse:
            loop, *_ = _make_loop(["/files"])
            loop.run()
            mock_parse.assert_not_called()


# ---------------------------------------------------------------------------
# Rendering sanitization — edit blocks wrapped in code fences
# ---------------------------------------------------------------------------

class TestRenderingSanitization:
    """Complete search/replace blocks in the LLM's response must be wrapped in
    triple-backtick fences before being passed to the Markdown renderer.
    Rich misinterprets <<<<<<< as HTML, ======= as a setext heading, and
    >>>>>>> as a blockquote, producing garbled output.
    """

    _BLOCK = (
        "<<<<<<< SEARCH foo.py\n"
        "old content\n"
        "=======\n"
        "new content\n"
        ">>>>>>> REPLACE"
    )

    def test_finish_stream_receives_fenced_text_when_block_present(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]):
                loop, _, _, _, renderer, _ = _make_loop(
                    ["hello"], stream_chunks=[self._BLOCK]
                )
                loop.run()
                _, rendered = renderer.finish_stream.call_args.args
                assert rendered.startswith("```")

    def test_finish_stream_fenced_text_preserves_block_content(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]):
                loop, _, _, _, renderer, _ = _make_loop(
                    ["hello"], stream_chunks=[self._BLOCK]
                )
                loop.run()
                _, rendered = renderer.finish_stream.call_args.args
                assert "<<<<<<< SEARCH foo.py" in rendered
                assert ">>>>>>> REPLACE" in rendered

    def test_finish_stream_plain_text_unchanged(self):
        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            loop, _, _, _, renderer, _ = _make_loop(
                ["hello"], stream_chunks=["just plain text"]
            )
            loop.run()
            _, rendered = renderer.finish_stream.call_args.args
            assert rendered == "just plain text"


# ---------------------------------------------------------------------------
# Proposal outcome feedback — LLM told whether edits were accepted/declined
# ---------------------------------------------------------------------------

class TestProposalOutcomeFeedback:
    """After every LLM turn that contained edit proposals, a user-role message
    is appended to session.history recording whether each edit was applied or
    declined.  This prevents the LLM from assuming a declined edit went through
    on subsequent turns.
    """

    def _outcome(self, name="test.py", applied=True):
        path = MagicMock()
        path.name = name
        return EditOutcome(path=path, applied=applied, snapshot=MagicMock() if applied else None)

    def test_no_proposals_no_feedback_in_history(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]):
            loop, session, *_ = _make_loop(["hello"])
            loop.run()
            # Only the real user message + assistant reply
            assert len(session.history) == 2

    def test_applied_feedback_added_and_says_applied(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits",
                       return_value=[self._outcome("test.py", applied=True)]):
                loop, session, *_ = _make_loop(["hello"])
                loop.run()
                assert len(session.history) == 3
                feedback = session.history[2]
                assert feedback["role"] == "user"
                assert "test.py" in feedback["content"]
                assert "applied" in feedback["content"].lower()

    def test_declined_feedback_says_not_applied(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits",
                       return_value=[self._outcome("test.py", applied=False)]):
                loop, session, *_ = _make_loop(["hello"])
                loop.run()
                feedback = session.history[2]["content"]
                assert "test.py" in feedback
                assert "not applied" in feedback.lower()

    def test_multiple_files_produce_single_feedback_message(self):
        outcomes = [self._outcome("foo.py", True), self._outcome("bar.py", True)]
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits", return_value=outcomes):
                loop, session, *_ = _make_loop(["hello"])
                loop.run()
                # user + assistant + ONE combined feedback message
                assert len(session.history) == 3
                feedback = session.history[2]["content"]
                assert "foo.py" in feedback
                assert "bar.py" in feedback

    def test_feedback_appears_in_subsequent_turn_messages(self):
        """Turn 2's LLM call must include turn 1's outcome feedback so the
        model knows the first edit was declined."""
        call_count = {"n": 0}

        def fake_parse(response, sess):
            call_count["n"] += 1
            return [MagicMock()] if call_count["n"] == 1 else []

        with patch("fizzy.chat_loop.file_context.refresh_files", return_value=[]):
            with patch("fizzy.chat_loop.edit_applier.parse_proposals", side_effect=fake_parse):
                with patch("fizzy.chat_loop.edit_applier.apply_edits",
                           return_value=[self._outcome("test.py", applied=False)]):
                    loop, _, client, *_ = _make_loop(["hello", "world"])
                    client.stream.side_effect = [iter(["reply1"]), iter(["reply2"])]
                    loop.run()

        second_call_msgs = client.stream.call_args_list[1].args[0]
        all_content = " ".join(m["content"] for m in second_call_msgs)
        assert "test.py" in all_content
        assert "not applied" in all_content.lower()


# ---------------------------------------------------------------------------
# Checkpoint recording — a turn's applied edits grouped into one checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointRecording:
    def _outcome(self, applied, snapshot=None):
        return EditOutcome(path=MagicMock(), applied=applied, snapshot=snapshot)

    def test_record_checkpoint_called_with_applied_snapshots(self):
        snap = MagicMock()
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits",
                       return_value=[self._outcome(True, snap)]):
                with patch("fizzy.chat_loop.change_history.record_checkpoint") as mock_cp:
                    loop, *_ = _make_loop(["hello"])
                    loop.run()
                    mock_cp.assert_called_once()
                    assert mock_cp.call_args.args[1] == [snap]

    def test_record_checkpoint_not_called_when_nothing_applied(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits",
                       return_value=[self._outcome(False, None)]):
                with patch("fizzy.chat_loop.change_history.record_checkpoint") as mock_cp:
                    loop, *_ = _make_loop(["hello"])
                    loop.run()
                    mock_cp.assert_not_called()

    def test_record_checkpoint_not_called_when_no_proposals(self):
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[]):
            with patch("fizzy.chat_loop.change_history.record_checkpoint") as mock_cp:
                loop, *_ = _make_loop(["hello"])
                loop.run()
                mock_cp.assert_not_called()

    def test_only_applied_snapshots_included_in_checkpoint(self):
        snap = MagicMock()
        outcomes = [self._outcome(True, snap), self._outcome(False, None)]
        with patch("fizzy.chat_loop.edit_applier.parse_proposals", return_value=[MagicMock()]):
            with patch("fizzy.chat_loop.edit_applier.apply_edits", return_value=outcomes):
                with patch("fizzy.chat_loop.change_history.record_checkpoint") as mock_cp:
                    loop, *_ = _make_loop(["hello"])
                    loop.run()
                    mock_cp.assert_called_once()
                    assert mock_cp.call_args.args[1] == [snap]


# ---------------------------------------------------------------------------
# /undo and /redo command dispatch
# ---------------------------------------------------------------------------

class TestUndoRedoCommands:
    def _checkpoint(self, *names):
        cp = MagicMock()
        snapshots = []
        for n in names:
            s = MagicMock()
            s.path.name = n
            snapshots.append(s)
        cp.snapshots = snapshots
        return cp

    # /undo

    def test_undo_does_not_call_stream(self):
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=None):
            loop, _, client, *_ = _make_loop(["/undo"])
            loop.run()
            client.stream.assert_not_called()

    def test_undo_calls_undo_last(self):
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=None) as mock_undo:
            loop, *_ = _make_loop(["/undo"])
            loop.run()
            mock_undo.assert_called_once()

    def test_undo_success_adds_history_note(self):
        cp = self._checkpoint("foo.py")
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=cp):
            loop, session, *_ = _make_loop(["/undo"])
            loop.run()
            assert len(session.history) == 1
            assert session.history[0]["role"] == "user"
            assert "foo.py" in session.history[0]["content"]

    def test_undo_success_note_lists_all_files(self):
        cp = self._checkpoint("a.py", "b.py")
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=cp):
            loop, session, *_ = _make_loop(["/undo"])
            loop.run()
            note = session.history[0]["content"]
            assert "a.py" in note and "b.py" in note

    def test_undo_noop_adds_no_history(self):
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=None):
            loop, session, *_ = _make_loop(["/undo"])
            loop.run()
            assert session.history == []

    def test_normal_message_after_undo_still_sent(self):
        with patch("fizzy.chat_loop.change_history.undo_last", return_value=None):
            loop, _, client, *_ = _make_loop(["/undo", "hello"])
            loop.run()
            client.stream.assert_called_once()

    # /redo

    def test_redo_does_not_call_stream(self):
        with patch("fizzy.chat_loop.change_history.redo_last", return_value=None):
            loop, _, client, *_ = _make_loop(["/redo"])
            loop.run()
            client.stream.assert_not_called()

    def test_redo_calls_redo_last(self):
        with patch("fizzy.chat_loop.change_history.redo_last", return_value=None) as mock_redo:
            loop, *_ = _make_loop(["/redo"])
            loop.run()
            mock_redo.assert_called_once()

    def test_redo_success_adds_history_note(self):
        cp = self._checkpoint("foo.py")
        with patch("fizzy.chat_loop.change_history.redo_last", return_value=cp):
            loop, session, *_ = _make_loop(["/redo"])
            loop.run()
            assert len(session.history) == 1
            assert "foo.py" in session.history[0]["content"]

    def test_redo_noop_adds_no_history(self):
        with patch("fizzy.chat_loop.change_history.redo_last", return_value=None):
            loop, session, *_ = _make_loop(["/redo"])
            loop.run()
            assert session.history == []


# ---------------------------------------------------------------------------
# Sequential same-file edits in a single turn must compose
# ---------------------------------------------------------------------------

class TestSequentialSameFileEdits:
    """When one assistant turn proposes two edits to the SAME file, both fold
    into one whole-file snapshot (one combined diff, one confirmation, one
    write).  The second edit builds on the first, and the single checkpoint
    undoes back to the original content.
    """

    def _block(self, filename: str, search: str, replace: str) -> str:
        return (
            f"<<<<<<< SEARCH {filename}\n"
            f"{search}"
            f"=======\n"
            f"{replace}"
            f">>>>>>> REPLACE"
        )

    def _loop_for(self, tmp_path, inputs, reply):
        """Build a ChatLoop rooted at a real tmp_path with a real LLM reply."""
        session = Session(
            model="claude-3-5-sonnet-20241022",
            working_dir=tmp_path,
            max_tokens=100_000,
        )
        client = MagicMock()
        client.stream.return_value = iter([reply])
        reader = MagicMock()
        reader.read_line.side_effect = list(inputs) + [EOFError]
        renderer = MagicMock()
        live_ctx = MagicMock()
        live_ctx.__enter__ = MagicMock(return_value=live_ctx)
        live_ctx.__exit__ = MagicMock(return_value=False)
        renderer.start_stream.return_value = live_ctx
        tracker = MagicMock()
        tracker.check.return_value = (TokenStatus.OK, 500)
        loop = ChatLoop(
            session=session, client=client, reader=reader,
            renderer=renderer, tracker=tracker,
        )
        return loop, session

    def _run_two_edit_turn(self, tmp_path):
        """One turn that edits two different regions of foo.py; both confirmed."""
        path = tmp_path / "foo.py"
        original = "line one\nline two\nline three\n"
        path.write_text(original, encoding="utf-8")
        resolved = path.resolve()
        reply = (
            self._block("foo.py", "line one\n", "LINE ONE\n")
            + "\n\n"
            + self._block("foo.py", "line three\n", "LINE THREE\n")
        )
        # inputs: user prompt, then ONE [y/N] confirmation for the combined file edit
        loop, session = self._loop_for(tmp_path, ["edit it", "y"], reply)
        session.context_files[resolved] = FileEntry(
            content=original, mtime=path.stat().st_mtime
        )
        loop.run()
        return session, path, original

    def test_both_edits_present_on_disk(self, tmp_path):
        _, path, _ = self._run_two_edit_turn(tmp_path)
        # The first edit (LINE ONE) must survive the second write.
        assert path.read_text() == "LINE ONE\nline two\nLINE THREE\n"

    def test_single_checkpoint_one_snapshot(self, tmp_path):
        session, *_ = self._run_two_edit_turn(tmp_path)
        assert len(session.undo_stack) == 1
        # both blocks combine into ONE whole-file snapshot for the file
        snaps = session.undo_stack[0].snapshots
        assert len(snaps) == 1
        assert snaps[0].before == "line one\nline two\nline three\n"
        assert snaps[0].after == "LINE ONE\nline two\nLINE THREE\n"

    def test_undo_restores_original_after_two_edits(self, tmp_path):
        session, path, original = self._run_two_edit_turn(tmp_path)
        # file is unchanged since the write (mtime matches) → undo applies directly
        reader = MagicMock()
        reader.read_line.return_value = "y"
        result = change_history.undo_last(session, MagicMock(), reader)
        assert result is not None
        assert path.read_text() == original
