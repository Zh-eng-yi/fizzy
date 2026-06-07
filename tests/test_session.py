from pathlib import Path

import pytest

from fizzy.session import Session


@pytest.fixture
def session():
    return Session(
        model="claude-3-5-sonnet-20241022",
        working_dir=Path("/tmp"),
        max_tokens=10_000,
    )


class TestAddMessages:
    def test_add_user_message(self, session):
        session.add_user_message("hello")
        assert session.history == [{"role": "user", "content": "hello"}]

    def test_add_assistant_message(self, session):
        session.add_assistant_message("hi")
        assert session.history == [{"role": "assistant", "content": "hi"}]

    def test_alternating_messages_preserve_order(self, session):
        session.add_user_message("ping")
        session.add_assistant_message("pong")
        session.add_user_message("ping again")
        assert [m["role"] for m in session.history] == ["user", "assistant", "user"]

    def test_empty_string_message_is_allowed(self, session):
        session.add_user_message("")
        assert session.history[0]["content"] == ""


class TestPopLastMessage:
    def test_pop_removes_last(self, session):
        session.add_user_message("a")
        session.add_assistant_message("b")
        popped = session.pop_last_message()
        assert popped == {"role": "assistant", "content": "b"}
        assert len(session.history) == 1

    def test_pop_on_empty_returns_none(self, session):
        assert session.pop_last_message() is None

    def test_pop_leaves_history_empty_when_one_item(self, session):
        session.add_user_message("only")
        session.pop_last_message()
        assert session.history == []


class TestClearHistory:
    def test_clear_empties_history(self, session):
        session.add_user_message("a")
        session.add_assistant_message("b")
        session.clear_history()
        assert session.history == []

    def test_clear_on_empty_is_safe(self, session):
        session.clear_history()
        assert session.history == []

    def test_can_add_after_clear(self, session):
        session.add_user_message("before")
        session.clear_history()
        session.add_user_message("after")
        assert len(session.history) == 1
        assert session.history[0]["content"] == "after"


class TestDefaults:
    def test_history_starts_empty(self, session):
        assert session.history == []

    def test_context_files_starts_empty(self, session):
        assert session.context_files == []

    def test_different_sessions_do_not_share_history(self):
        """Mutable default field — each instance must get its own list."""
        s1 = Session(model="m", working_dir=Path("/"), max_tokens=1000)
        s2 = Session(model="m", working_dir=Path("/"), max_tokens=1000)
        s1.add_user_message("only in s1")
        assert s2.history == []
