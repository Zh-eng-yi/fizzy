from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    """Snapshot of a single file held in context: its text content and the
    mtime at the time it was last read.  Stored in Session.context_files."""

    content: str
    mtime: float  # os.stat().st_mtime at last read


@dataclass
class FileSnapshot:
    """One file's whole-file before/after contents for a single agent turn.

    ``before`` is the file content prior to the turn's edits (the undo target);
    ``after`` is the content the agent wrote (the redo target).  ``mtime`` is the
    ``os.stat().st_mtime`` of whichever state the tool last wrote for this file —
    updated on every undo/redo write — so divergence (an out-of-band user edit)
    can be detected cheaply, falling back to a content comparison.  Created by
    fizzy.edit_applier and grouped into checkpoints by fizzy.change_history."""

    path: Path     # resolved absolute path of the edited file
    before: str    # whole-file content before the turn's edits
    after: str     # whole-file content the agent wrote
    mtime: float   # mtime of the state the tool last wrote for this file


@dataclass
class Checkpoint:
    """One agent turn's applied edits, grouped as a single undo/redo unit.

    ``snapshots`` holds one FileSnapshot per changed file.  /undo reverts every
    file to its ``before``; /redo restores every ``after`` — atomically, with one
    confirmation.  Maintained by fizzy.change_history (Week 2)."""

    snapshots: list[FileSnapshot]


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
