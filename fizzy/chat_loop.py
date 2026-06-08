from fizzy import file_context
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

            # 3. Handle slash commands (no LLM call for any of these)
            if text.lower() in _EXIT_COMMANDS:
                self._renderer.print_info("Goodbye!")
                break

            if text.startswith("/add"):
                parts = text.split()
                if len(parts) != 2:
                    self._renderer.print_info("Usage: /add <file>")
                else:
                    ok, msg = file_context.add_file(self._session, parts[1])
                    if ok:
                        self._renderer.print_info(msg)
                    else:
                        self._renderer.print_error(msg)
                continue

            if text.startswith("/drop"):
                parts = text.split()
                if len(parts) != 2:
                    self._renderer.print_info("Usage: /drop <file>")
                else:
                    ok, msg = file_context.drop_file(self._session, parts[1])
                    if ok:
                        self._renderer.print_info(msg)
                    else:
                        self._renderer.print_error(msg)
                continue

            if text.strip() == "/files":
                paths = file_context.list_files(self._session)
                if not paths:
                    self._renderer.print_info("No files in context.")
                else:
                    lines = []
                    for path in paths:
                        try:
                            display = path.relative_to(self._session.working_dir)
                        except ValueError:
                            display = path
                        lines.append(str(display))
                    self._renderer.print_info("\n".join(lines))
                continue

            # 4. Append user message to history
            self._session.add_user_message(text)

            # 5. Refresh stale context files and print any change notices
            for notice in file_context.refresh_files(self._session):
                self._renderer.print_info(notice)

            # 6. Assemble the full message list for this turn.
            #    The system message (file contents) is built fresh and prepended;
            #    it is never stored in session.history.
            system_msg = file_context.build_system_message(self._session)
            messages = (
                [system_msg, *self._session.history]
                if system_msg
                else self._session.history
            )

            # 7. Check token budget before sending (counts file content too)
            status, used = self._tracker.check(messages)

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

            # 8. Stream the response
            accumulated = ""
            try:
                with self._renderer.start_stream() as live:
                    for chunk in self._client.stream(messages):
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

            # 9. Persist the assistant reply
            self._session.add_assistant_message(accumulated)

            # 10. Show token status (rebuild messages to include the assistant reply)
            final_messages = (
                [system_msg, *self._session.history]
                if system_msg
                else self._session.history
            )
            final_status, final_used = self._tracker.check(final_messages)
            self._tracker.render_status(final_status, final_used)
