from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Session:
    model: str
    working_dir: Path
    max_tokens: int
    history: list[dict] = field(default_factory=list)
    context_files: list[Path] = field(default_factory=list)

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
