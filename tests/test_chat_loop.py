from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from fizzy.chat_loop import ChatLoop
from fizzy.session import Session
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
