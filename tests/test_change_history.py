from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fizzy import change_history
from fizzy.session import Checkpoint, FileSnapshot, Session


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


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p.resolve()


def _checkpoint(*snaps: FileSnapshot) -> Checkpoint:
    return Checkpoint(snapshots=list(snaps))


def _clean_snapshot(tmp_path: Path, name: str, before: str, after: str):
    """A file currently holding `after`, with a recorded mtime that matches disk
    (i.e. the file is unchanged since the tool last wrote it)."""
    path = _write(tmp_path, name, after)
    return path, FileSnapshot(path=path, before=before, after=after, mtime=path.stat().st_mtime)


def _stale_snapshot(path: Path, before: str, after: str) -> FileSnapshot:
    """A snapshot whose recorded mtime cannot match any real file (mtime=0.0),
    forcing the content-comparison fallback to decide divergence."""
    return FileSnapshot(path=path, before=before, after=after, mtime=0.0)


# ── data model ────────────────────────────────────────────────────────────────


class TestFileSnapshot:
    def test_stores_fields(self, tmp_path):
        p = (tmp_path / "f.py").resolve()
        snap = FileSnapshot(path=p, before="a\n", after="b\n", mtime=123.0)
        assert (snap.path, snap.before, snap.after, snap.mtime) == (p, "a\n", "b\n", 123.0)


class TestCheckpoint:
    def test_stores_snapshots(self, tmp_path):
        snap = FileSnapshot(path=tmp_path / "f.py", before="a\n", after="b\n", mtime=1.0)
        cp = Checkpoint(snapshots=[snap])
        assert cp.snapshots == [snap]


# ── record_checkpoint ───────────────────────────────────────────────────────────


class TestRecordCheckpoint:
    def _snap(self, tmp_path, name="f.py"):
        return FileSnapshot(path=tmp_path / name, before="old\n", after="new\n", mtime=1.0)

    def test_empty_returns_none(self, session):
        assert change_history.record_checkpoint(session, []) is None

    def test_empty_does_not_push(self, session):
        change_history.record_checkpoint(session, [])
        assert session.undo_stack == []

    def test_pushes_one_checkpoint(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._snap(tmp_path)])
        assert len(session.undo_stack) == 1

    def test_returns_pushed_checkpoint(self, session, tmp_path):
        cp = change_history.record_checkpoint(session, [self._snap(tmp_path)])
        assert cp is session.undo_stack[-1]

    def test_holds_snapshots_in_order(self, session, tmp_path):
        snaps = [self._snap(tmp_path, "a.py"), self._snap(tmp_path, "b.py")]
        cp = change_history.record_checkpoint(session, snaps)
        assert [s.path.name for s in cp.snapshots] == ["a.py", "b.py"]

    def test_clears_redo_stack(self, session, tmp_path):
        session.redo_stack.append(_checkpoint(self._snap(tmp_path, "stale.py")))
        change_history.record_checkpoint(session, [self._snap(tmp_path)])
        assert session.redo_stack == []

    def test_multiple_checkpoints_stack_in_order(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._snap(tmp_path, "a.py")])
        change_history.record_checkpoint(session, [self._snap(tmp_path, "b.py")])
        assert [cp.snapshots[0].path.name for cp in session.undo_stack] == ["a.py", "b.py"]

    def test_stores_independent_copy(self, session, tmp_path):
        snaps = [self._snap(tmp_path, "a.py")]
        cp = change_history.record_checkpoint(session, snaps)
        snaps.append(self._snap(tmp_path, "b.py"))  # mutate caller's list afterwards
        assert len(cp.snapshots) == 1

    def test_does_not_mutate_history(self, session, tmp_path):
        change_history.record_checkpoint(session, [self._snap(tmp_path)])
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

    # unchanged file (mtime matches) → revert directly, no prompt

    def test_unchanged_reverts_to_before(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert path.read_text() == "OLD\n"

    def test_unchanged_does_not_prompt(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        reader = _mock_reader("n")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_not_called()

    def test_unchanged_returns_checkpoint(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        cp = _checkpoint(snap)
        session.undo_stack.append(cp)
        assert change_history.undo_last(session, _mock_renderer(), _mock_reader("n")) is cp

    def test_unchanged_moves_checkpoint_to_redo(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        cp = _checkpoint(snap)
        session.undo_stack.append(cp)
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert session.undo_stack == []
        assert session.redo_stack == [cp]

    def test_unchanged_prints_message(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        renderer = _mock_renderer()
        change_history.undo_last(session, renderer, _mock_reader("n"))
        renderer.print_info.assert_called()

    def test_updates_snapshot_mtime_after_write(self, session, tmp_path):
        # After writing `before`, the recorded mtime tracks the new on-disk state
        # so a later redo can detect divergence correctly.
        p1, s1 = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        p2, s2 = _clean_snapshot(tmp_path, "foo.py", before="NEW\n", after="NEWER\n")
        session.undo_stack.append(_checkpoint(s1))
        session.undo_stack.append(_checkpoint(s2))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert s2.mtime == p2.stat().st_mtime
        assert s1.mtime == p2.stat().st_mtime
    
    # mtime differs but content unchanged → still treated as unchanged

    def test_mtime_differs_content_same_no_prompt(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "NEW\n")          # content still equals `after`
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        reader = _mock_reader("n")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_not_called()
        assert path.read_text() == "OLD\n"

    # diverged (content actually changed) → confirmation required

    def test_diverged_prompts(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        reader = _mock_reader("y")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_called()

    def test_diverged_shows_diff(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        renderer = _mock_renderer()
        change_history.undo_last(session, renderer, _mock_reader("y"))
        renderer.print_info.assert_called()

    def test_diverged_confirmed_overwrites_to_before(self, session, tmp_path):
        # The user's out-of-band change is overwritten — no merge (MVP behavior).
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert path.read_text() == "OLD\n"

    @pytest.mark.parametrize("answer", ["n", "N", "no", "", "nope"])
    def test_diverged_declined_leaves_file_and_stacks(self, session, tmp_path, answer):
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        cp = _checkpoint(snap)
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader(answer))
        assert result is None
        assert path.read_text() == "USER_CHANGED\n"
        assert session.undo_stack == [cp]
        assert session.redo_stack == []

    # deleted file counts as divergence

    def test_deleted_file_prompts(self, session, tmp_path):
        gone = (tmp_path / "gone.py").resolve()  # never created on disk
        snap = _stale_snapshot(gone, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        reader = _mock_reader("y")
        change_history.undo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_called()

    def test_deleted_file_confirmed_recreates_before(self, session, tmp_path):
        gone = (tmp_path / "gone.py").resolve()
        snap = _stale_snapshot(gone, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("y"))
        assert gone.read_text() == "OLD\n"

    def test_deleted_file_declined_not_recreated(self, session, tmp_path):
        gone = (tmp_path / "gone.py").resolve()
        snap = _stale_snapshot(gone, before="OLD\n", after="NEW\n")
        session.undo_stack.append(_checkpoint(snap))
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert not gone.exists()

    # multi-file — atomic

    def test_multifile_unchanged_reverts_all_directly(self, session, tmp_path):
        pa, sa = _clean_snapshot(tmp_path, "a.py", "A0\n", "A1\n")
        pb, sb = _clean_snapshot(tmp_path, "b.py", "B0\n", "B1\n")
        session.undo_stack.append(_checkpoint(sa, sb))
        reader = _mock_reader("n")
        change_history.undo_last(session, _mock_renderer(), reader)
        assert pa.read_text() == "A0\n"
        assert pb.read_text() == "B0\n"
        reader.read_line.assert_not_called()

    def test_multifile_one_diverged_single_prompt_confirm_reverts_all(self, session, tmp_path):
        pa, sa = _clean_snapshot(tmp_path, "a.py", "A0\n", "A1\n")   # unchanged
        pb = _write(tmp_path, "b.py", "B_USER\n")                    # diverged
        sb = _stale_snapshot(pb, before="B0\n", after="B1\n")
        session.undo_stack.append(_checkpoint(sa, sb))
        reader = _mock_reader("y")
        change_history.undo_last(session, _mock_renderer(), reader)
        assert reader.read_line.call_count == 1                      # one atomic prompt
        assert pa.read_text() == "A0\n"
        assert pb.read_text() == "B0\n"

    def test_multifile_diverged_declined_writes_nothing(self, session, tmp_path):
        pa, sa = _clean_snapshot(tmp_path, "a.py", "A0\n", "A1\n")
        pb = _write(tmp_path, "b.py", "B_USER\n")
        sb = _stale_snapshot(pb, before="B0\n", after="B1\n")
        cp = _checkpoint(sa, sb)
        session.undo_stack.append(cp)
        result = change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert result is None
        assert pa.read_text() == "A1\n"          # untouched
        assert pb.read_text() == "B_USER\n"
        assert session.undo_stack == [cp]


# ── redo_last ───────────────────────────────────────────────────────────────────


class TestRedoLast:
    def test_empty_stack_returns_none(self, session):
        assert change_history.redo_last(session, _mock_renderer(), _mock_reader()) is None

    def test_empty_stack_prints_info(self, session):
        renderer = _mock_renderer()
        change_history.redo_last(session, renderer, _mock_reader())
        renderer.print_info.assert_called()

    # for redo the file is in the UNDONE state: it holds `before`

    def test_unchanged_reapplies_after(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "OLD\n")
        snap = FileSnapshot(path=path, before="OLD\n", after="NEW\n", mtime=path.stat().st_mtime)
        session.redo_stack.append(_checkpoint(snap))
        reader = _mock_reader("n")
        change_history.redo_last(session, _mock_renderer(), reader)
        assert path.read_text() == "NEW\n"
        reader.read_line.assert_not_called()

    def test_unchanged_moves_checkpoint_to_undo(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "OLD\n")
        snap = FileSnapshot(path=path, before="OLD\n", after="NEW\n", mtime=path.stat().st_mtime)
        cp = _checkpoint(snap)
        session.redo_stack.append(cp)
        change_history.redo_last(session, _mock_renderer(), _mock_reader("n"))
        assert session.redo_stack == []
        assert session.undo_stack == [cp]
    
    def test_updates_snapshot_mtime_after_write(self, session, tmp_path):
        # After writing `after`, the recorded mtime tracks the new on-disk state
        # so a later undo can detect divergence correctly.
        p1, s1 = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        p2, s2 = _clean_snapshot(tmp_path, "foo.py", before="NEW\n", after="NEWER\n")
        session.redo_stack.append(_checkpoint(s2))
        session.redo_stack.append(_checkpoint(s1))
        change_history.redo_last(session, _mock_renderer(), _mock_reader("n"))
        assert s1.mtime == p1.stat().st_mtime
        assert s2.mtime == p1.stat().st_mtime

    def test_diverged_prompts_and_overwrites_to_after(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        session.redo_stack.append(_checkpoint(snap))
        reader = _mock_reader("y")
        change_history.redo_last(session, _mock_renderer(), reader)
        reader.read_line.assert_called()
        assert path.read_text() == "NEW\n"

    def test_diverged_declined_leaves_file(self, session, tmp_path):
        path = _write(tmp_path, "foo.py", "USER_CHANGED\n")
        snap = _stale_snapshot(path, before="OLD\n", after="NEW\n")
        cp = _checkpoint(snap)
        session.redo_stack.append(cp)
        result = change_history.redo_last(session, _mock_renderer(), _mock_reader("n"))
        assert result is None
        assert path.read_text() == "USER_CHANGED\n"
        assert session.redo_stack == [cp]


# ── undo/redo round trip ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_undo_then_redo_restores_edit(self, session, tmp_path):
        path, snap = _clean_snapshot(tmp_path, "foo.py", before="OLD\n", after="NEW\n")
        cp = _checkpoint(snap)
        session.undo_stack.append(cp)

        # unchanged → both directions apply directly (no prompt needed)
        change_history.undo_last(session, _mock_renderer(), _mock_reader("n"))
        assert path.read_text() == "OLD\n"
        assert session.redo_stack == [cp]

        change_history.redo_last(session, _mock_renderer(), _mock_reader("n"))
        assert path.read_text() == "NEW\n"
        assert session.undo_stack == [cp]
        assert session.redo_stack == []
