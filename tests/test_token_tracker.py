from pathlib import Path
from unittest.mock import patch

import pytest

from fizzy.session import Session
from fizzy.token_tracker import TokenStatus, TokenTracker, _BLOCK_THRESHOLD, _WARN_THRESHOLD


@pytest.fixture
def session():
    return Session(
        model="claude-3-5-sonnet-20241022",
        working_dir=Path("/tmp"),
        max_tokens=10_000,
    )


@pytest.fixture
def tracker(session):
    return TokenTracker(session=session)


HISTORY = [{"role": "user", "content": "hello"}]


class TestCount:
    def test_returns_litellm_token_count(self, tracker):
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=42):
            assert tracker.count(HISTORY) == 42

    def test_returns_zero_on_any_error(self, tracker):
        with patch("fizzy.token_tracker.litellm.token_counter", side_effect=Exception("fail")):
            assert tracker.count(HISTORY) == 0

    def test_passes_correct_args_to_litellm(self, tracker, session):
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=5) as mock_fn:
            tracker.count(HISTORY)
        mock_fn.assert_called_once_with(
            model=session.model,
            messages=HISTORY,
        )

    def test_empty_history_returns_zero(self, tracker):
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=0):
            assert tracker.count([]) == 0


class TestCheck:
    def test_ok_status_below_warn_threshold(self, tracker, session):
        used = int(session.max_tokens * (_WARN_THRESHOLD - 0.01))
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=used):
            status, returned_used = tracker.check(HISTORY)
        assert status == TokenStatus.OK
        assert returned_used == used

    def test_warn_status_at_warn_threshold(self, tracker, session):
        used = int(session.max_tokens * _WARN_THRESHOLD)
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=used):
            status, _ = tracker.check(HISTORY)
        assert status == TokenStatus.WARN

    def test_warn_status_between_thresholds(self, tracker, session):
        used = int(session.max_tokens * ((_WARN_THRESHOLD + _BLOCK_THRESHOLD) / 2))
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=used):
            status, _ = tracker.check(HISTORY)
        assert status == TokenStatus.WARN

    def test_block_status_at_block_threshold(self, tracker, session):
        used = int(session.max_tokens * _BLOCK_THRESHOLD)
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=used):
            status, _ = tracker.check(HISTORY)
        assert status == TokenStatus.BLOCK

    def test_block_status_above_max(self, tracker, session):
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=session.max_tokens + 1):
            status, _ = tracker.check(HISTORY)
        assert status == TokenStatus.BLOCK

    def test_zero_tokens_is_ok(self, tracker):
        with patch("fizzy.token_tracker.litellm.token_counter", return_value=0):
            status, used = tracker.check(HISTORY)
        assert status == TokenStatus.OK
        assert used == 0

    def test_counter_error_degrades_to_ok(self, tracker):
        """If counting fails, used=0 which is safely OK — no false block."""
        with patch("fizzy.token_tracker.litellm.token_counter", side_effect=Exception("fail")):
            status, used = tracker.check(HISTORY)
        assert status == TokenStatus.OK
        assert used == 0


class TestRenderStatus:
    def test_render_does_not_raise_for_any_status(self, tracker):
        for status in TokenStatus:
            tracker.render_status(status, used=500)

    def test_render_at_max_tokens_does_not_raise(self, tracker, session):
        tracker.render_status(TokenStatus.BLOCK, used=session.max_tokens)
