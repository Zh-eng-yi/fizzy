from unittest.mock import MagicMock, patch

import litellm.exceptions as llm_exc
import pytest

from fizzy.llm_client import LLMClient


@pytest.fixture
def client():
    return LLMClient(api_key="test-key", model="claude-3-5-sonnet-20241022")


HISTORY = [{"role": "user", "content": "hello"}]


def _make_completion_response(chunks: list[str]):
    """Build a mock litellm streaming response that yields the given text chunks."""
    def _iter():
        for text in chunks:
            chunk = MagicMock()
            chunk.choices[0].delta.content = text
            yield chunk

    return _iter()


class TestStream:
    def test_yields_chunks_in_order(self, client):
        chunks = ["Hello", ", ", "world", "!"]
        with patch("fizzy.llm_client.litellm.completion", return_value=_make_completion_response(chunks)):
            assert list(client.stream(HISTORY)) == chunks

    def test_empty_response_yields_nothing(self, client):
        with patch("fizzy.llm_client.litellm.completion", return_value=_make_completion_response([])):
            assert list(client.stream(HISTORY)) == []

    def test_none_delta_content_is_skipped(self, client):
        """Chunks with None content (e.g. role-only deltas) must be silently dropped."""
        def _iter():
            for content in [None, "hello", None, "!"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = content
                yield chunk

        with patch("fizzy.llm_client.litellm.completion", return_value=_iter()):
            assert list(client.stream(HISTORY)) == ["hello", "!"]

    def test_passes_correct_args_to_litellm(self, client):
        with patch("fizzy.llm_client.litellm.completion", return_value=_make_completion_response(["ok"])) as mock_call:
            list(client.stream(HISTORY))
        mock_call.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            messages=HISTORY,
            api_key="test-key",
            stream=True,
        )

    def test_model_stored_on_client(self, client):
        assert client.model == "claude-3-5-sonnet-20241022"


class TestRetryOnRateLimit:
    def test_retries_up_to_three_times_then_raises(self, client):
        exc = llm_exc.RateLimitError("rate limited", llm_provider="anthropic", model="x", response=MagicMock())
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="retries exhausted"):
                list(client.stream(HISTORY))

    def test_succeeds_after_transient_rate_limit(self, client):
        exc = llm_exc.RateLimitError("rl", llm_provider="anthropic", model="x", response=MagicMock())
        responses = [exc, _make_completion_response(["recovered"])]
        with patch("fizzy.llm_client.litellm.completion", side_effect=responses), \
             patch("time.sleep"):
            assert list(client.stream(HISTORY)) == ["recovered"]

    def test_sleep_called_between_retries(self, client):
        exc = llm_exc.RateLimitError("rl", llm_provider="anthropic", model="x", response=MagicMock())
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                list(client.stream(HISTORY))
        assert mock_sleep.call_count == 2  # 3 attempts → 2 sleeps

    def test_daily_quota_exhausted_raises_immediately(self, client):
        exc = llm_exc.RateLimitError(
            "limit: 0 PerDay quota", llm_provider="gemini", model="x", response=MagicMock()
        )
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Daily free-tier quota exhausted"):
                list(client.stream(HISTORY))
        mock_sleep.assert_not_called()


class TestAuthenticationError:
    def test_raises_runtime_error_immediately(self, client):
        exc = llm_exc.AuthenticationError("bad key", llm_provider="anthropic", model="x", response=MagicMock())
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc):
            with pytest.raises(RuntimeError, match="Authentication failed"):
                list(client.stream(HISTORY))

    def test_no_retry_on_auth_error(self, client):
        exc = llm_exc.AuthenticationError("bad key", llm_provider="anthropic", model="x", response=MagicMock())
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc) as mock_call, \
             patch("time.sleep"):
            with pytest.raises(RuntimeError):
                list(client.stream(HISTORY))
        assert mock_call.call_count == 1


class TestConnectionError:
    def test_retries_on_api_connection_error(self, client):
        exc = llm_exc.APIConnectionError("timeout", llm_provider="anthropic", model="x")
        responses = [exc, _make_completion_response(["ok"])]
        with patch("fizzy.llm_client.litellm.completion", side_effect=responses), \
             patch("time.sleep"):
            assert list(client.stream(HISTORY)) == ["ok"]

    def test_raises_after_max_connection_retries(self, client):
        exc = llm_exc.APIConnectionError("timeout", llm_provider="anthropic", model="x")
        with patch("fizzy.llm_client.litellm.completion", side_effect=exc), \
             patch("time.sleep"):
            with pytest.raises(llm_exc.APIConnectionError):
                list(client.stream(HISTORY))
