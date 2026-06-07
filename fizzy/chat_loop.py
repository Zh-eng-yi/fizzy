from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.llm_client import LLMClient
from fizzy.session import Session
from fizzy.token_tracker import TokenStatus, TokenTracker

_EXIT_COMMANDS = {"/quit", "/exit", "/q"}


class ChatLoop:
    def __init__(
        self,
        session: Session,
        client: LLMClient,
        reader: InputReader,
        renderer: OutputRenderer,
        tracker: TokenTracker,
    ) -> None:
        self._session = session
        self._client = client
        self._reader = reader
        self._renderer = renderer
        self._tracker = tracker

    def run(self) -> None:
        self._renderer.print_info(
            f"fizzy — model: {self._session.model}  |  "
            f"context: {self._session.max_tokens:,} tokens  |  "
            "Ctrl-D to exit"
        )

        while True:
            # 1. Read input
            try:
                text = self._reader.read_line("You: ")
            except EOFError:
                self._renderer.print_info("\nGoodbye!")
                break

            # 2. Skip empty / whitespace-only input
            text = text.strip()
            if not text:
                continue

            # 3. Handle exit commands
            if text.lower() in _EXIT_COMMANDS:
                self._renderer.print_info("Goodbye!")
                break

            # 4. Append user message to history
            self._session.add_user_message(text)

            # 5. Check token budget before sending
            status, used = self._tracker.check(self._session.history)

            if status == TokenStatus.BLOCK:
                self._renderer.print_error(
                    f"Context window is full ({used:,} / "
                    f"{self._session.max_tokens:,} tokens). "
                    "Cannot send — please start a new session."
                )
                self._session.pop_last_message()
                continue

            if status == TokenStatus.WARN:
                self._renderer.print_info(
                    f"Warning: context is {used / self._session.max_tokens:.0%} full "
                    f"({used:,} / {self._session.max_tokens:,} tokens)."
                )

            # 6. Stream the response
            accumulated = ""
            try:
                with self._renderer.start_stream() as live:
                    for chunk in self._client.stream(self._session.history):
                        accumulated += chunk
                        self._renderer.append_chunk(live, accumulated)
                    self._renderer.finish_stream(live, accumulated)
            except RuntimeError as exc:
                # Auth errors and other fatal client errors
                self._renderer.print_error(str(exc))
                self._session.pop_last_message()
                continue
            except Exception as exc:
                self._renderer.print_error(f"Unexpected error: {exc}")
                self._session.pop_last_message()
                continue

            # 7. Persist the assistant reply
            self._session.add_assistant_message(accumulated)

            # 8. Show token status
            final_status, final_used = self._tracker.check(self._session.history)
            self._tracker.render_status(final_status, final_used)
