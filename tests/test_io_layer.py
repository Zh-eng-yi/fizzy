"""
Tests for io_layer.py — written against requirements, not implementation.

InputReader requirements:
  - Returns the user's input as a stripped string
  - Raises EOFError when the user signals end-of-input (Ctrl-D)
  - Returns an empty string when the user interrupts (Ctrl-C)
  - Persists input history to a file across calls

OutputRenderer requirements:
  - print_error: writes a message to the console that includes the given text
  - print_info:  writes a message to the console that includes the given text
  - start_stream / append_chunk / finish_stream: manage a live-updating display;
      finish_stream must always produce the final complete text and a blank separator
"""
from io import StringIO
from unittest.mock import MagicMock, patch, call

import pytest
from rich.console import Console

from fizzy.io_layer import InputReader, OutputRenderer


# ---------------------------------------------------------------------------
# InputReader
# ---------------------------------------------------------------------------

class TestInputReaderReturnsInput:
    def test_returns_stripped_text(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.return_value = "  hello world  "
            reader = InputReader()
            assert reader.read_line() == "hello world"

    def test_returns_empty_string_for_blank_input(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.return_value = ""
            reader = InputReader()
            assert reader.read_line() == ""

    def test_passes_prompt_string_to_underlying_session(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.return_value = "x"
            reader = InputReader()
            reader.read_line("Ask me: ")
            MockSession.return_value.prompt.assert_called_once_with("Ask me: ")

    def test_default_prompt_is_you(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.return_value = "x"
            reader = InputReader()
            reader.read_line()
            prompt_arg = MockSession.return_value.prompt.call_args.args[0]
            assert "You" in prompt_arg


class TestInputReaderCtrlD:
    def test_raises_eof_error_on_ctrl_d(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.side_effect = EOFError
            reader = InputReader()
            with pytest.raises(EOFError):
                reader.read_line()


class TestInputReaderCtrlC:
    def test_returns_empty_string_on_ctrl_c(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.side_effect = KeyboardInterrupt
            reader = InputReader()
            assert reader.read_line() == ""

    def test_does_not_raise_on_ctrl_c(self):
        with patch("fizzy.io_layer.PromptSession") as MockSession:
            MockSession.return_value.prompt.side_effect = KeyboardInterrupt
            reader = InputReader()
            try:
                reader.read_line()
            except KeyboardInterrupt:
                pytest.fail("KeyboardInterrupt was not caught by InputReader")


class TestInputReaderHistory:
    def test_history_file_is_configured(self):
        """History must be backed by a file, not in-memory only."""
        with patch("fizzy.io_layer.PromptSession") as MockSession, \
             patch("fizzy.io_layer.FileHistory") as MockFileHistory:
            MockSession.return_value.prompt.return_value = ""
            InputReader()
            MockFileHistory.assert_called_once()
            path_arg = str(MockFileHistory.call_args.args[0])
            assert "fizzy" in path_arg  # history file is fizzy-specific


# ---------------------------------------------------------------------------
# OutputRenderer
# ---------------------------------------------------------------------------

def _renderer_with_captured_console():
    """Return an OutputRenderer whose console writes to a StringIO buffer."""
    buf = StringIO()
    console = Console(file=buf, highlight=False, markup=False)
    renderer = OutputRenderer()
    renderer._console = console
    return renderer, buf


class TestPrintError:
    def test_output_contains_message_text(self):
        renderer, buf = _renderer_with_captured_console()
        renderer.print_error("something went wrong")
        assert "something went wrong" in buf.getvalue()

    def test_different_messages_are_reflected(self):
        renderer, buf = _renderer_with_captured_console()
        renderer.print_error("auth failed")
        assert "auth failed" in buf.getvalue()

    def test_empty_message_does_not_raise(self):
        renderer, buf = _renderer_with_captured_console()
        renderer.print_error("")  # must not raise


class TestPrintInfo:
    def test_output_contains_message_text(self):
        renderer, buf = _renderer_with_captured_console()
        renderer.print_info("context is 80% full")
        assert "context is 80% full" in buf.getvalue()

    def test_empty_message_does_not_raise(self):
        renderer, buf = _renderer_with_captured_console()
        renderer.print_info("")


class TestStream:
    def _make_mock_live(self):
        live = MagicMock()
        live.__enter__ = MagicMock(return_value=live)
        live.__exit__ = MagicMock(return_value=False)
        return live

    def test_start_stream_returns_context_manager(self):
        renderer, _ = _renderer_with_captured_console()
        live = renderer.start_stream()
        # Must support the context manager protocol
        assert hasattr(live, "__enter__") and hasattr(live, "__exit__")

    def test_finish_stream_final_text_is_shown(self):
        """finish_stream must display the full accumulated text."""
        renderer, _ = _renderer_with_captured_console()
        live = self._make_mock_live()
        renderer.finish_stream(live, "final response text")
        # The live display was updated at least once with the final content
        assert live.update.called
        last_call_arg = live.update.call_args.args[0]
        # The renderable passed to update must encode the final text
        assert "final response text" in last_call_arg.markup

    def test_finish_stream_stops_live(self):
        renderer, _ = _renderer_with_captured_console()
        live = self._make_mock_live()
        renderer.finish_stream(live, "done")
        live.stop.assert_called_once()

    def test_finish_stream_prints_blank_separator(self):
        """A blank line must be printed after the stream ends."""
        renderer, buf = _renderer_with_captured_console()
        live = self._make_mock_live()
        renderer.finish_stream(live, "done")
        assert "\n" in buf.getvalue()

    def test_append_chunk_updates_live_eventually(self):
        """After enough chunks, the live display must be updated."""
        renderer, _ = _renderer_with_captured_console()
        live = self._make_mock_live()
        # Send more than one screen-worth of chunks
        for i in range(50):
            renderer.append_chunk(live, "x" * i)
        assert live.update.called

    def test_finish_stream_after_no_chunks_does_not_raise(self):
        renderer, _ = _renderer_with_captured_console()
        live = self._make_mock_live()
        renderer.finish_stream(live, "")
