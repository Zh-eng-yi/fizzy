from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fizzy import change_history
from fizzy.session import ChangeRecord, Checkpoint, Session


# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def session(tmp_path):
    return Session(model="test-model", working_dir=tmp_path, max_tokens=100_000)


def _mock_renderer():
    return MagicMock()


def _mock_reader(response: str = "y"):
    reader = MagicMock()
    reader.read_line.return_value = response
    return reader


def _file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a file with post-edit content and return its resolved path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p.resolve()


def _checkpoint(*records: ChangeRecord) -> Checkpoint:
    return Checkpoint(records=list(records))


# ── data model ──────────────────────────────────────────────────────────────────


class TestChangeRecord:
    def test_stores_path_search_replace(self, tmp_path):
        p = (tmp_path / "f.py").resolve()
        rec = ChangeRecord(path=p, search="old\n", replace="new\n")
        assert (rec.path, rec.search, rec.replace) == (p, "old\n", "new\n")


class TestCheckpoint:
    def test_stores_records(self, tmp_path):
        rec = ChangeRecord(path=tmp_path / "f.py", search="a\n", replace="b\n")
        cp = Checkpoint(records=[rec])
        assert cp.records == [rec]


# ── record_checkpoint ───────────────────────────────────────────────────────────


class TestRecordCheckpoint:
    def _rec(self, tmp_path, name="f.py"):
        return ChangeRecord(path=tmp_path / name, search="a\n", replace="b\n")

    def test_empty_records_returns_none(self, session):
        assert change_history.record_checkpoint(session, []) is None

    def test_empty_records_does_not_push(self, session):
        change_history.record_checkpoint(session, [])
        assert session.undo_stack == []

    def test_pushes_one_checkpoint(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._rec(tmp_path)])
        assert len(session.undo_stack) == 1

    def test_returns_pushed_checkpoint(self, session, tmp_path):
        cp = change_history.record_checkpoint(session, [self._rec(tmp_path)])
        assert cp is session.undo_stack[-1]

    def test_checkpoint_holds_records_in_order(self, session, tmp_path):
        recs = [self._rec(tmp_path, "a.py"), self._rec(tmp_path, "b.py")]
        cp = change_history.record_checkpoint(session, recs)
        assert [r.path.name for r in cp.records] == ["a.py", "b.py"]

    def test_clears_redo_stack(self, session, tmp_path):
        session.redo_stack.append(_checkpoint(self._rec(tmp_path, "stale.py")))
        change_history.record_checkpoint(session, [self._rec(tmp_path)])
        assert session.redo_stack == []

    def test_multiple_checkpoints_stack_in_order(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._rec(tmp_path, "a.py")])
        change_history.record_checkpoint(session, [self._rec(tmp_path, "b.py")])
        assert [cp.records[0].path.name for cp in session.undo_stack] == ["a.py", "b.py"]

    def test_stores_independent_copy(self, session, tmp_path):
        recs = [self._rec(tmp_path, "a.py")]
        cp = change_history.record_checkpoint(session, recs)
        recs.append(self._rec(tmp_path, "b.py"))  # mutate caller's list afterwards
        assert len(cp.records) == 1

    def test_does_not_mutate_history(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._rec(tmp_path)])
        assert session.history == []


# ── undo_last ───────────────────────────────────────────────────────────────────


class TestUndoLast:
    # empty stack

    def test_empty_stack_returns_none(self, session):
        assert change_history.undo_last(session, _mock_renderer(), _mock_reader()) is None

    def test_empty_stack_prints_info(self, session):
        renderer = _mock_renderer()
        change_history.undo_last(session, renderer, _mock_reader())
        renderer.print_info.assert_called()

    def test_empty_stack_does_not_prompt(self, session):
        reader = _mock_reader()
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_not_called()

    # happy path — inverse edit restores the original

    def test_reverts_the_edit(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nnew_func()\nb\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "a\nold_func()\nb\n"

    def test_returns_checkpoint(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        assert change_history.undo_last(session, _mock_renderer(), _mock_reader("y")) is cp

    def test_moves_checkpoint_to_redo(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert session.undo_stack == []
        assert session.redo_stack == [cp]

    def test_preserves_unrelated_user_edits(self, session, tmp_path):
        # The user changed an unrelated line after the agent's edit; undo must
        # revert only the agent's region and keep the user's edit.
        path = _file(tmp_path, "foo.py", "a\nnew_func()\nUSER_EDIT\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "a\nold_func()\nUSER_EDIT\n"

    def test_handles_file_without_trailing_newline(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()")  # no trailing newline
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "old_func()"

    # confirmation / display

    def test_shows_diff_before_prompt(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        renderer = _mock_renderer()
        change_history.undo_last(session, renderer, _mock_reader("y"))
        renderer.print_info.assert_called()

    def test_prompts_for_confirmation(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        reader = _mock_reader("y")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_called()

    @pytest.mark.parametrize("answer", ["n", "N", "no", "", "nope"])
    def test_declined_leaves_file_unchanged(self, session, tmp_path, answer):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader(answer))
        assert result is None
        assert (tmp_path / "foo.py").read_text() == "new_func()\n"

    def test_declined_leaves_stacks_untouched(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert session.undo_stack == [cp]
        assert session.redo_stack == []

    # conflicts — the recorded change can no longer be located

    def test_conflict_when_region_changed_returns_none(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nTOTALLY_DIFFERENT\nb\n")  # replace text gone
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert result is None

    def test_conflict_leaves_file_unchanged(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nTOTALLY_DIFFERENT\nb\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "a\nTOTALLY_DIFFERENT\nb\n"

    def test_conflict_keeps_checkpoint_on_undo_stack(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nTOTALLY_DIFFERENT\nb\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert session.undo_stack == [cp]
        assert session.redo_stack == []

    def test_conflict_prints_error(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nTOTALLY_DIFFERENT\nb\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        renderer = _mock_renderer()
        change_history.undo_last(session, renderer, _mock_reader("y"))
        renderer.print_error.assert_called()

    def test_conflict_does_not_prompt(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nTOTALLY_DIFFERENT\nb\n")
        session.undo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        reader = _mock_reader("y")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_not_called()

    def test_ambiguous_replace_returns_none(self, session, tmp_path):
        # The replacement text now appears twice — undo cannot pick one safely.
        path = _file(tmp_path, "foo.py", "new_func()\nnew_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert result is None
        assert (tmp_path / "foo.py").read_text() == "new_func()\nnew_func()\n"

    def test_missing_file_returns_none(self, session, tmp_path):
        gone = (tmp_path / "gone.py").resolve()  # never created on disk
        cp = _checkpoint(ChangeRecord(path=gone, search="old\n", replace="new\n"))
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert result is None
        assert session.undo_stack == [cp]

    # multi-file checkpoints

    def test_multifile_reverts_all(self, session, tmp_path):
        a = _file(tmp_path, "a.py", "A_new\n")
        b = _file(tmp_path, "b.py", "B_new\n")
        session.undo_stack.append(_checkpoint(
            ChangeRecord(path=a, search="A_old\n", replace="A_new\n"),
            ChangeRecord(path=b, search="B_old\n", replace="B_new\n"),
        ))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "a.py").read_text() == "A_old\n"
        assert (tmp_path / "b.py").read_text() == "B_old\n"

    def test_multifile_atomic_when_one_conflicts(self, session, tmp_path):
        # b.py conflicts → nothing is written, including the clean a.py.
        a = _file(tmp_path, "a.py", "A_new\n")
        b = _file(tmp_path, "b.py", "B_CHANGED\n")  # B_new no longer present
        cp = _checkpoint(
            ChangeRecord(path=a, search="A_old\n", replace="A_new\n"),
            ChangeRecord(path=b, search="B_old\n", replace="B_new\n"),
        )
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert result is None
        assert (tmp_path / "a.py").read_text() == "A_new\n"      # NOT reverted
        assert (tmp_path / "b.py").read_text() == "B_CHANGED\n"
        assert session.undo_stack == [cp]

    def test_same_file_two_edits_revert_in_reverse_order(self, session, tmp_path):
        # edit1 v0→v1, edit2 v1→v2; file currently v2; undo must end at v0.
        path = _file(tmp_path, "foo.py", "v2\n")
        session.undo_stack.append(_checkpoint(
            ChangeRecord(path=path, search="v0\n", replace="v1\n"),
            ChangeRecord(path=path, search="v1\n", replace="v2\n"),
        ))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "v0\n"


# ── redo_last ───────────────────────────────────────────────────────────────────


class TestRedoLast:
    def test_empty_stack_returns_none(self, session):
        assert change_history.redo_last(session, _mock_renderer(), _mock_reader()) is None

    def test_empty_stack_prints_info(self, session):
        renderer = _mock_renderer()
        change_history.redo_last(session, renderer, _mock_reader())
        renderer.print_info.assert_called()

    def test_reapplies_edit(self, session, tmp_path):
        # File is in the undone state (contains the original text).
        path = _file(tmp_path, "foo.py", "old_func()\n")
        session.redo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "new_func()\n"

    def test_moves_checkpoint_to_undo(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "old_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.redo_stack.append(cp)
        change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert session.redo_stack == []
        assert session.undo_stack == [cp]

    def test_preserves_unrelated_user_edits(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "a\nold_func()\nUSER_EDIT\n")
        session.redo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "a\nnew_func()\nUSER_EDIT\n"

    def test_conflict_returns_none(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "SOMETHING_ELSE\n")  # search text gone
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.redo_stack.append(cp)
        result = change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert result is None
        assert (tmp_path / "foo.py").read_text() == "SOMETHING_ELSE\n"
        assert session.redo_stack == [cp]

    def test_declined_leaves_file_unchanged(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "old_func()\n")
        session.redo_stack.append(
            _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        )
        result = change_history.redo_last(session, _mock_renderer(), _mock_reader("n"))
        assert result is None
        assert (tmp_path / "foo.py").read_text() == "old_func()\n"

    def test_same_file_two_edits_reapply_in_forward_order(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "v0\n")
        session.redo_stack.append(_checkpoint(
            ChangeRecord(path=path, search="v0\n", replace="v1\n"),
            ChangeRecord(path=path, search="v1\n", replace="v2\n"),
        ))
        change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "v2\n"


# ── undo/redo round trip ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_undo_then_redo_restores_edit(self, session, tmp_path):
        path = _file(tmp_path, "foo.py", "new_func()\n")
        cp = _checkpoint(ChangeRecord(path=path, search="old_func()\n", replace="new_func()\n"))
        session.undo_stack.append(cp)

        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "old_func()\n"

        change_history.redo_last(session, _mock_renderer(), _mock_reader("y"))
        assert (tmp_path / "foo.py").read_text() == "new_func()\n"
        assert session.undo_stack == [cp]
        assert session.redo_stack == []
