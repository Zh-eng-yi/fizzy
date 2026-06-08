from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from fizzy.chat_loop import ChatLoop
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
            assert "UNIQUE_MARKER = True" in messages_sent[0]["content"]

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
