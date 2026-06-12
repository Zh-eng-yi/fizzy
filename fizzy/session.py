from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    """Snapshot of a single file held in context: its text content and the
    mtime at the time it was last read.  Stored in Session.context_files."""

    content: str
    mtime: float  # os.stat().st_mtime at last read


@dataclass
class ChangeRecord:
    """One applied search/replace edit, stored as an inverse-applicable fragment.

    ``search``/``replace`` are the *effective* texts that were applied (after the
    narrow trailing-newline fallback), widened with surrounding context where
    needed so that each is uniquely locatable.  Undo reverses the edit (replace
    ``replace`` with ``search``); redo re-applies it (``search`` → ``replace``).
    Created by fizzy.edit_applier and grouped into checkpoints by
    fizzy.change_history (Week 2)."""

    path: Path     # resolved absolute path of the edited file
    search: str    # text that was replaced (unique in the pre-edit file)
    replace: str   # text it became (unique in the post-edit file)


@dataclass
class Checkpoint:
    """One agent turn's applied edits, grouped as a single undo/redo unit.

    ``records`` are in application order.  /undo reverses the whole checkpoint;
    /redo re-applies it.  Maintained by fizzy.change_history (Week 2)."""

    records: list[ChangeRecord]


@dataclass
class Session:
    model: str
    working_dir: Path
    max_tokens: int
    history: list[dict] = field(default_factory=list)
    # Insertion-ordered map of resolved Path → FileEntry.
    # Populated and maintained by fizzy.file_context (Week 2).
    context_files: dict[Path, FileEntry] = field(default_factory=dict)
    # Undo/redo stacks of checkpoints, maintained by fizzy.change_history
    # (Week 2).  A turn's applied edits are grouped into one Checkpoint pushed
    # onto undo_stack (which clears redo_stack); /undo and /redo move whole
    # checkpoints between the two stacks.
    undo_stack: list[Checkpoint] = field(default_factory=list)
    redo_stack: list[Checkpoint] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def pop_last_message(self) -> dict | None:
        """Remove and return the last message (used to roll back a blocked send)."""
        return self.history.pop() if self.history else None

    def clear_history(self) -> None:
        """Clear all message history (for future context compaction)."""
        self.history.clear()
