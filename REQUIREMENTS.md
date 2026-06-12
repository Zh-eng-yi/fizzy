# One-Month Work Plan: Terminal AI Agent

## Guiding Principles

- Build vertically (end-to-end thin slice) before going wide
- Each week produces something runnable
- Defer polish, config options, and edge cases
- Every component that touches the filesystem or shell requires a confirmation gate
- Token budget is a first-class concern from day one, not a Week 4 afterthought

---

## Week 1 — Minimal Working Loop

**Goal:** You can type a message, it reaches an LLM, and the response streams back to your terminal.

| Component | Responsibility |
|---|---|
| Entry point | Reads CLI arguments (model, API key, working dir), bootstraps and connects all components, starts the session |
| LLM client | Takes a list of messages → streams back response tokens. Handles retries and auth errors. |
| I/O layer | Takes user input with line-editing and history. Renders streamed output with code block formatting. |
| Chat loop | Maintains the message history list. Feeds input to the LLM client, appends response, loops. |
| Session state | Holds in-memory state: current files in context, message history, working directory |
| Token tracker | Counts tokens in the message history on every turn and surfaces the current usage. Emits a warning when approaching the model's context limit. Does not yet compress — just tracks and warns. |

**End of week:** A working chat REPL. No file editing yet.

**Testing:** Manual smoke test — send several long exchanges and verify the token tracker warning fires before the API rejects the request.

**Risk note:** Context limits will be hit in real sessions long before Week 5. The token tracker is intentionally minimal here but must be in place so Week 5's compressor has something to build on.

---

## Week 2 — File Context & Editing

**Goal:** Add files to the conversation and have the LLM edit them — with a human confirmation gate before any change touches disk.

| Component | Responsibility |
|---|---|
| File context manager | Tracks which files are part of the current conversation and makes their contents available to the LLM |
| Edit applier | Interprets the LLM's proposed changes, renders a diff for the user to review, and only writes to disk after explicit approval |
| Confirmation gate | Before any disk write: shows a colored diff, prompts yes/no/edit. Blocks the edit applier from proceeding without approval. |
| Slash commands | `/add <file>`, `/drop <file>`, `/files`, `/undo`, `/redo`. Each command mutates session state or triggers an action. |
| Change history | Records what each agent turn changed so edits can be reviewed, undone, and redone without relying on git — and without discarding unrelated edits the user made in the meantime |

**End of week:** You can ask the LLM to edit a real file, review the diff, approve it, and see the change applied. `/undo` reverts the last change and `/redo` re-applies it.

**Testing:** Unit-test the edit applier against known before/after file pairs. Verify that refusing confirmation leaves the file untouched. Verify `/undo` restores the original content and `/redo` re-applies it, and that an unrelated edit elsewhere in the file is preserved across undo.

---

## Week 3 — Repo Awareness

**Goal:** The agent has useful knowledge of files not explicitly added to the conversation.

| Component | Responsibility |
|---|---|
| Codebase indexer | Scans the repo and builds a queryable representation of its structure and symbols. Stays up to date as files change. |
| Context selector | Takes the current conversation state → decides which parts of the index are most relevant → returns a summary small enough to fit in the prompt |

**End of week:** The agent can reference and reason about code outside the files explicitly added to the chat.

**Testing:** Ask the agent about a symbol defined in a file not yet `/add`-ed. Verify it surfaces the correct location without hallucinating.

---

## Week 4 — Agentic Loop

**Goal:** The agent can autonomously plan and execute multi-step tasks — reading files, making edits, running checks, and retrying — without the user driving each step.

| Component | Responsibility |
|---|---|
| Tool registry | Defines the set of tools the agent may call: read file, write file (via confirmation gate), run lint, run shell command (via confirmation gate), web fetch. Each tool has a name, description, and input/output schema exposed to the LLM. |
| Planner | Given a high-level task, the LLM produces a step-by-step plan before executing. The plan is shown to the user for approval before the agent begins. |
| Tool dispatcher | Parses the LLM's tool-call output and routes each call to the correct tool implementation. Returns results back to the LLM for the next step. |
| Agentic loop | Drives the plan → tool-call → observation → next-step cycle until the task is complete or the agent declares it stuck. Surfaces intermediate results at each step. |
| Loop guardrails | Caps the maximum number of autonomous steps per task (configurable). If the cap is hit, pauses and asks the user how to proceed rather than looping indefinitely. |

**`/run` command safety:** Shell command execution is a tool available to the agentic loop. It requires the same confirmation gate as file edits: the command is shown to the user before it runs, stdout/stderr is captured, and the output is optionally appended to the chat context. The agent must never construct and run a shell command from unsanitized user input without displaying it first.

**End of week:** You can say "refactor all usages of `foo()` to `bar()` across the repo, run the linter, and fix any errors it finds" and the agent executes it step by step, pausing for confirmation at each destructive action.

**Testing:** Run a multi-step task with a deliberate error in step 2. Verify the agent catches it, retries, and does not loop past the step cap.

---

## Week 5 — Polish & Power Features

**Goal:** Make it feel like a real tool, not a prototype.

| Component | Responsibility |
|---|---|
| Context window manager | Builds on the Week 1 token tracker. When approaching the model's limit, summarizes older messages to compress history while preserving recent context and all file snapshots. |
| Linter feedback loop | After edits are applied, runs a syntax/lint check on changed files. If errors are found, feeds them back to the LLM for a self-correction pass (via the agentic loop). |
| `/web` command | Fetches a URL, strips it to plain text, injects the content into the chat context. Available as a tool in the agentic loop. |
| Multi-model support | `/model <name>` switches the active model mid-session. Maintains a registry of model metadata: context window size, token cost, capability flags. |
| Config file | Loads defaults (model, API key, token budget, working dir) from a file in the project root or home directory at startup |
| Git integration | After confirmed edits: offers to stage and commit changes. Shows the user the branch name and diff before staging. Refuses to auto-commit on a dirty working tree — prompts the user to stash or resolve conflicts first. Feeds the diff to the LLM to generate a commit message, which the user can edit before confirming. |

**End of week:** A genuinely useful daily-driver tool.

**Testing:** Simulate a context-limit scenario with a long synthetic history. Verify the compressor reduces token count while the agent retains knowledge of edits made earlier in the session.

---

## Cross-Cutting Concerns (apply every week)

| Concern | Policy |
|---|---|
| Confirmation before disk writes | Every file edit and shell execution shows the user what will happen and requires explicit approval. No exceptions. |
| Token budget awareness | Every LLM call checks remaining budget before sending. Warn at 80%, block and compress at 95%. |
| Undo safety net | Change history (Week 2) is independent of git — it works even in non-git directories. |
| Step cap on agentic loop | Default maximum of 20 autonomous steps per task. Configurable. Always surfaces progress to the user between steps. |
| No silent failures | If a tool call fails (lint error, file not found, API error), the failure is shown to the user and added to the LLM context. The agent does not silently skip it. |
