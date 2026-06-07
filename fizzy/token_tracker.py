from enum import Enum, auto

import litellm
from rich.console import Console
from rich.text import Text

from fizzy.session import Session

_WARN_THRESHOLD = 0.80
_BLOCK_THRESHOLD = 0.95


class TokenStatus(Enum):
    OK = auto()
    WARN = auto()   # >= 80% of max_tokens
    BLOCK = auto()  # >= 95% of max_tokens


class TokenTracker:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._console = Console()

    def count(self, history: list[dict]) -> int:
        """Return the token count for the given history.

        Uses litellm.token_counter, which is a local calculation —
        no API call is made. Falls back to 0 on any unexpected error.
        """
        try:
            return litellm.token_counter(
                model=self._session.model,
                messages=history,
            )
        except Exception:
            return 0

    def check(self, history: list[dict]) -> tuple[TokenStatus, int]:
        """Return (status, used_tokens) for the current history."""
        used = self.count(history)
        ratio = used / self._session.max_tokens
        if ratio >= _BLOCK_THRESHOLD:
            status = TokenStatus.BLOCK
        elif ratio >= _WARN_THRESHOLD:
            status = TokenStatus.WARN
        else:
            status = TokenStatus.OK
        return status, used

    def render_status(self, status: TokenStatus, used: int) -> None:
        """Print a compact token usage bar after each turn."""
        max_t = self._session.max_tokens
        pct = used / max_t

        color = {
            TokenStatus.OK: "green",
            TokenStatus.WARN: "yellow",
            TokenStatus.BLOCK: "red",
        }[status]

        label = Text(f"Tokens: {used:,} / {max_t:,} ({pct:.0%})", style=color)
        self._console.print(label)
