"""Record applied file edits as checkpoints and undo/redo them without git.

State lives in ``Session.undo_stack`` / ``Session.redo_stack`` (lists of
Checkpoints); this module is a stateless helper, mirroring fizzy.file_context.
Undo/redo apply the *inverse* search/replace edit to the current file, so an
unrelated change elsewhere in the file is preserved — only a genuine edit to the
same region blocks the operation.  The undo/redo split follows the classic
editor model: recording a new checkpoint clears the redo stack.

Public API
----------
record_checkpoint -- group a turn's edits into one Checkpoint; clear redo
undo_last         -- reverse the latest checkpoint, with confirmation
redo_last         -- re-apply the most recently undone checkpoint
"""

import difflib
from pathlib import Path

from fizzy import edit_applier
from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.session import ChangeRecord, Checkpoint, Session


def record_checkpoint(
    session: Session, records: list[ChangeRecord]
) -> Checkpoint | None:
    """Group a turn's applied edits into one Checkpoint.

    Pushes the Checkpoint onto the undo stack and clears the redo stack (a fresh
    edit invalidates any redo history).  Returns the Checkpoint, or None when
    there is nothing to record.
    """
    if not records:
        return None
    checkpoint = Checkpoint(records=list(records))
    session.undo_stack.append(checkpoint)
    session.redo_stack.clear()
    return checkpoint


def _read_current(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _diff(path: Path, old: str, new: str) -> str:
    name = str(path)
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=name,
            tofile=name,
        )
    )


def _apply_checkpoint(
    checkpoint: Checkpoint,
    renderer: OutputRenderer,
    reader: InputReader,
    *,
    reverse: bool,
    verb: str,
) -> Checkpoint | None:
    """Shared engine for undo (reverse=True) and redo (reverse=False).

    Dry-runs the inverse (or forward) edit for every file on in-memory working
    copies and aborts without writing if any file conflicts.  Otherwise shows
    the per-file diff, asks once for confirmation, writes every file, and returns
    the checkpoint.  Stack movement is left to the callers.
    """
    records = list(reversed(checkpoint.records)) if reverse else list(checkpoint.records)

    original: dict[Path, str] = {}
    working: dict[Path, str] = {}
    for record in records:
        if record.path not in working:
            current = _read_current(record.path)
            if current is None:
                renderer.print_error(
                    f"Cannot {verb} '{record.path.name}': file is missing or unreadable."
                )
                return None
            original[record.path] = current
            working[record.path] = current

        if reverse:
            find, sub = record.replace, record.search
        else:
            find, sub = record.search, record.replace
        result = edit_applier.try_replace(working[record.path], find, sub)
        if result.content is None:
            renderer.print_error(
                f"Cannot {verb} '{record.path.name}': {result.error} — "
                "the relevant lines were edited since; resolve manually."
            )
            return None
        working[record.path] = result.content

    for path in working:
        renderer.print_info(_diff(path, original[path], working[path]))

    names = ", ".join(sorted({r.path.name for r in checkpoint.records}))
    response = reader.read_line(
        f"{verb.capitalize()} the last change ({len(working)} file(s): {names})? [y/N]: "
    )
    if response.strip().lower() not in ("y", "yes"):
        return None

    for path, content in working.items():
        path.write_text(content, encoding="utf-8")
    return checkpoint


def undo_last(
    session: Session, renderer: OutputRenderer, reader: InputReader
) -> Checkpoint | None:
    """Reverse the most recent checkpoint via inverse edits, with confirmation.

    Returns the undone Checkpoint on success; None when there is nothing to undo,
    a file conflicts, or the user declines.
    """
    if not session.undo_stack:
        renderer.print_info("Nothing to undo.")
        return None
    checkpoint = session.undo_stack[-1]
    result = _apply_checkpoint(checkpoint, renderer, reader, reverse=True, verb="undo")
    if result is not None:
        session.undo_stack.pop()
        session.redo_stack.append(checkpoint)
    return result


def redo_last(
    session: Session, renderer: OutputRenderer, reader: InputReader
) -> Checkpoint | None:
    """Re-apply the most recently undone checkpoint, with confirmation.

    Returns the re-applied Checkpoint on success; None when there is nothing to
    redo, a file conflicts, or the user declines.
    """
    if not session.redo_stack:
        renderer.print_info("Nothing to redo.")
        return None
    checkpoint = session.redo_stack[-1]
    result = _apply_checkpoint(checkpoint, renderer, reader, reverse=False, verb="redo")
    if result is not None:
        session.redo_stack.pop()
        session.undo_stack.append(checkpoint)
    return result
