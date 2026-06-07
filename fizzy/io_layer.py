from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

_HISTORY_FILE = Path.home() / ".fizzy_history"
_RERENDER_EVERY = 20  # characters between live re-renders


class InputReader:
    def __init__(self) -> None:
        self._session: PromptSession = PromptSession(
            history=FileHistory(str(_HISTORY_FILE))
        )

    def read_line(self, prompt_str: str = "You: ") -> str:
        """Block for user input and return the stripped text.

        Raises EOFError on Ctrl-D (signals clean exit).
        Returns "" on Ctrl-C (caller should continue the loop).
        """
        try:
            return self._session.prompt(prompt_str).strip()
        except KeyboardInterrupt:
            return ""


class OutputRenderer:
    def __init__(self) -> None:
        self._console = Console()
        self._char_count = 0

    def start_stream(self) -> Live:
        """Return a rich Live context manager. Enter it with `with`."""
        self._char_count = 0
        return Live(
            Markdown(""),
            console=self._console,
            refresh_per_second=10,
            vertical_overflow="visible",
        )

    def append_chunk(self, live: Live, accumulated: str) -> None:
        """Update the live display with the latest accumulated text.

        Re-renders only every _RERENDER_EVERY new characters to reduce CPU load.
        """
        self._char_count += 1
        if self._char_count % _RERENDER_EVERY == 0:
            live.update(Markdown(accumulated))

    def finish_stream(self, live: Live, accumulated: str) -> None:
        """Finalize the live display with the complete response."""
        live.update(Markdown(accumulated))
        live.stop()
        self._console.print()  # blank separator line

    def print_error(self, msg: str) -> None:
        self._console.print(f"[bold red]Error:[/bold red] {msg}")

    def print_info(self, msg: str) -> None:
        self._console.print(f"[dim]{msg}[/dim]")

    def print_panel(self, renderable, title: str = "") -> None:
        self._console.print(Panel(renderable, title=title))
