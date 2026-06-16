"""Record applied file edits as checkpoints and undo/redo them without git.

State lives in ``Session.undo_stack`` / ``Session.redo_stack`` (lists of
Checkpoints); this module is a stateless helper, mirroring fizzy.file_context.

Each checkpoint stores, per changed file, a whole-file ``before``/``after``
FileSnapshot.  Undo reverts every file to its ``before``; redo restores every
``after`` — the file is treated as a unit and overwritten wholesale.  Undo/redo
are **atomic**: one confirmation reverts the whole checkpoint, or nothing.

Out-of-band edits (the user changed a file outside fizzy since we last wrote it)
are detected by mtime, falling back to a content comparison when the mtime
differs.  When a file has diverged the diff is shown and a single confirmation is
required before the overwrite (no 3-way merge — MVP).  When nothing diverged the
operation applies directly with a brief message.

Public API
----------
record_checkpoint -- group a turn's snapshots into one Checkpoint; clear redo
undo_last         -- revert the latest checkpoint to its before-state
redo_last         -- re-apply the most recently undone checkpoint's after-state
"""

from pathlib import Path

from fizzy.edit_applier import compute_diff
from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.session import Checkpoint, FileSnapshot, Session


def record_checkpoint(
    session: Session, snapshots: list[FileSnapshot]
) -> Checkpoint | None:
    """Group a turn's file snapshots into one Checkpoint.

    Pushes the Checkpoint onto the undo stack and clears the redo stack (a fresh
    edit invalidates any redo history).  Returns the Checkpoint, or None when
    there is nothing to record.
    """
    if not snapshots:
        return None
    checkpoint = Checkpoint(snapshots=list(snapshots))
    session.undo_stack.append(checkpoint)
    session.redo_stack.clear()
    return checkpoint


def _read_current(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _sync_mtime(session: Session, path: Path, mtime: float) -> None:
    """Record *mtime* as the last-tool-write mtime for *path* on every snapshot
    of that file across both stacks.

    There is one file on disk per path, so all snapshots referring to it share a
    single "last written by us" mtime.  Keeping them in sync means a later
    undo/redo of a *different* checkpoint touching the same file still detects
    out-of-band edits correctly.
    """
    for checkpoint in (*session.undo_stack, *session.redo_stack):
        for snap in checkpoint.snapshots:
            if snap.path == path:
                snap.mtime = mtime


def _last_write_mtime(session: Session, path: Path) -> float | None:
    """The most recent mtime the tool recorded for *path* across both stacks.

    Every snapshot of a file shares a single "last written by us" mtime (kept in
    sync by _sync_mtime after each write), so the latest recorded value is the
    moment we last touched the file — what divergence is measured against.
    """
    mtimes = [
        snap.mtime
        for checkpoint in (*session.undo_stack, *session.redo_stack)
        for snap in checkpoint.snapshots
        if snap.path == path
    ]
    return max(mtimes) if mtimes else None


def _has_diverged(
    session: Session, snapshot: FileSnapshot, current: str | None, expected: str
) -> bool:
    """True if the file changed outside fizzy since we last wrote it.

    A missing file counts as diverged.  Otherwise mtime is the fast path: if the
    on-disk mtime still matches the path's last-write mtime the file is in a
    tool-known state (unchanged).  If it differs we fall back to comparing the
    on-disk content against the state we expect it to be in before this operation
    (``after`` for undo, ``before`` for redo).
    """
    if current is None:
        return True
    try:
        disk_mtime = snapshot.path.stat().st_mtime
    except OSError:
        return True
    if disk_mtime == _last_write_mtime(session, snapshot.path):
        return False
    return current != expected


def _apply_checkpoint(
    session: Session,
    checkpoint: Checkpoint,
    renderer: OutputRenderer,
    reader: InputReader,
    *,
    attr: str,
    verb: str,
) -> Checkpoint | None:
    """Shared engine for undo (attr="before") and redo (attr="after").

    Reads each file's current content, decides whether any file diverged, and —
    if so — shows the diffs and asks one atomic ``[y/N]``.  On confirm (or when
    nothing diverged) every file is overwritten with its target state and the
    recorded mtimes are re-synced.  Returns the checkpoint, or None when the user
    declines.  Stack movement is left to the callers.
    """
    # The file is expected to currently hold the state opposite the target:
    # undo (target=before) expects `after`; redo (target=after) expects `before`.
    expected_attr = "after" if attr == "before" else "before"

    plans: list[tuple[FileSnapshot, str | None, str]] = []
    diverged = False
    for snap in checkpoint.snapshots:
        current = _read_current(snap.path)
        target = getattr(snap, attr)
        if _has_diverged(session, snap, current, getattr(snap, expected_attr)):
            diverged = True
        plans.append((snap, current, target))

    names = ", ".join(sorted({s.path.name for s in checkpoint.snapshots}))
    if diverged:
        for snap, current, target in plans:
            renderer.print_info(compute_diff(snap.path, current or "", target))
        renderer.print_info(
            f"Warning: {verb} will overwrite changes made to these files outside fizzy."
        )
        response = reader.read_line(
            f"{verb.capitalize()} {len(plans)} file(s) ({names})? [y/N]: "
        )
        if response.strip().lower() not in ("y", "yes"):
            return None
    else:
        renderer.print_info(f"{verb.capitalize()} {len(plans)} file(s): {names}.")

    for snap, _current, target in plans:
        snap.path.write_text(target, encoding="utf-8")
        _sync_mtime(session, snap.path, snap.path.stat().st_mtime)
    return checkpoint


def undo_last(
    session: Session, renderer: OutputRenderer, reader: InputReader
) -> Checkpoint | None:
    """Revert the most recent checkpoint to each file's ``before`` content.

    Returns the undone Checkpoint on success; None when there is nothing to undo
    or the user declines (after a divergence warning).
    """
    if not session.undo_stack:
        renderer.print_info("Nothing to undo.")
        return None
    checkpoint = session.undo_stack[-1]
    result = _apply_checkpoint(
        session, checkpoint, renderer, reader, attr="before", verb="undo"
    )
    if result is not None:
        session.undo_stack.pop()
        session.redo_stack.append(checkpoint)
    return result


def redo_last(
    session: Session, renderer: OutputRenderer, reader: InputReader
) -> Checkpoint | None:
    """Re-apply the most recently undone checkpoint's ``after`` content.

    Returns the re-applied Checkpoint on success; None when there is nothing to
    redo or the user declines (after a divergence warning).
    """
    if not session.redo_stack:
        renderer.print_info("Nothing to redo.")
        return None
    checkpoint = session.redo_stack[-1]
    result = _apply_checkpoint(
        session, checkpoint, renderer, reader, attr="after", verb="redo"
    )
    if result is not None:
        session.redo_stack.pop()
        session.undo_stack.append(checkpoint)
    return result
