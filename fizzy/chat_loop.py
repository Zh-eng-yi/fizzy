from fizzy import change_history, edit_applier, file_context
from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.llm_client import LLMClient
from fizzy.prompts import AGENT_INSTRUCTIONS
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

            if text == "/undo":
                checkpoint = change_history.undo_last(
                    self._session, self._renderer, self._reader
                )
                if checkpoint is not None:
                    names = ", ".join(sorted({s.path.name for s in checkpoint.snapshots}))
                    self._session.add_user_message(
                        f"Reverted the last change ({names})."
                    )
                continue

            if text == "/redo":
                checkpoint = change_history.redo_last(
                    self._session, self._renderer, self._reader
                )
                if checkpoint is not None:
                    names = ", ".join(sorted({s.path.name for s in checkpoint.snapshots}))
                    self._session.add_user_message(
                        f"Re-applied the last undone change ({names})."
                    )
                continue

            # 4. Append user message to history
            self._session.add_user_message(text)

            # 5. Refresh stale context files and print any change notices
            for notice in file_context.refresh_files(self._session):
                self._renderer.print_info(notice)

            # 6. Assemble the full message list for this turn.
            #    When files are in context, two system messages are prepended:
            #      [0] agent instructions (edit format rules)
            #      [1] file contents (built fresh from context_files)
            #    Neither is ever stored in session.history.
            system_msg = file_context.build_system_message(self._session)
            if system_msg:
                instructions_msg = {"role": "system", "content": AGENT_INSTRUCTIONS}
                messages = [instructions_msg, system_msg, *self._session.history]
            else:
                messages = self._session.history

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
                        self._renderer.append_chunk(
                            live,
                            edit_applier.sanitize_for_display(accumulated),
                        )
                    self._renderer.finish_stream(
                        live,
                        edit_applier.sanitize_for_display(accumulated),
                    )
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

            # 9.5. Process any edit proposals in the reply and record outcomes.
            #      apply_edits groups proposals by file, folds each file's blocks
            #      into one combined whole-file change, and asks a single atomic
            #      confirmation for the turn.  The per-file outcome is written to
            #      session.history so the LLM knows next turn whether each edit was
            #      applied or declined (preventing context drift).  Applied files'
            #      before/after snapshots are grouped into one undo checkpoint.
            proposals = edit_applier.parse_proposals(accumulated, self._session)
            if proposals:
                edit_outcomes = edit_applier.apply_edits(
                    proposals, self._session, self._renderer, self._reader
                )
                snapshots = [o.snapshot for o in edit_outcomes if o.applied]
                if snapshots:
                    change_history.record_checkpoint(self._session, snapshots)
                self._session.add_user_message(
                    "\n".join(
                        f"Edit to '{o.path.name}': "
                        f"{'applied' if o.applied else 'not applied — file unchanged'}."
                        for o in edit_outcomes
                    )
                )

            # 10. Show token status (rebuild messages to include the assistant reply)
            if system_msg:
                instructions_msg = {"role": "system", "content": AGENT_INSTRUCTIONS}
                final_messages = [instructions_msg, system_msg, *self._session.history]
            else:
                final_messages = self._session.history
            final_status, final_used = self._tracker.check(final_messages)
            self._tracker.render_status(final_status, final_used)
