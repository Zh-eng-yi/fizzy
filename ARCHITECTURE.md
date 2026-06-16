# Architecture

## Overview

fizzy is a terminal AI agent built as a Python REPL. The user types messages into a prompt, each message is sent to an LLM via [litellm](https://github.com/BerriAI/litellm), and the response streams back with Markdown and syntax-highlighted code blocks. The design is intentionally thin — each component has one responsibility, and the chat loop is the only place that coordinates between them.

```
User input
    │
    ▼
┌─────────────┐     messages      ┌────────────┐     litellm     ┌──────────────┐
│ InputReader │ ────────────────► │  ChatLoop  │ ──────────────► │  LLMClient   │
└─────────────┘                   │            │ ◄────────────── └──────────────┘
                                  │            │   token stream
┌──────────────┐   render chunks  │            │
│OutputRenderer│ ◄────────────────│            │
└──────────────┘                  │            │
                                  │            │──── read/write ──► Session
┌──────────────┐   check/render   │            │
│ TokenTracker │ ◄────────────────│            │
└──────────────┘                  └────────────┘
```

---

## Components

### `session.py` — Session State

The single source of truth for all in-memory state. Passed by reference to every component that needs it; nothing else holds its own copy of history.

```
Session
  model: str                          # litellm model string (e.g. "gemini/gemini-2.5-flash")
  working_dir: Path
  max_tokens: int                     # context window limit for the active model
  history: list[dict]                 # Anthropic/OpenAI message format: [{role, content}, ...]
  context_files: dict[Path, FileEntry]  # insertion-ordered; managed by file_context.py
  undo_stack: list[Checkpoint]        # applied-edit checkpoints (LIFO); managed by change_history.py
  redo_stack: list[Checkpoint]        # checkpoints that were undone and may be redone
```

`FileEntry` (defined in `session.py`): `content: str`, `mtime: float` — the file's text and the `os.stat().st_mtime` at last read.

`FileSnapshot` (defined in `session.py`): `path: Path`, `before: str`, `after: str`, `mtime: float` — one file's whole-file content before and after a turn's edits. `before` is the undo target, `after` the redo target; the file is treated as a unit (overwritten wholesale, not patched). `mtime` is the stat-mtime of whichever state the tool last wrote for this file (updated on every undo/redo write) so out-of-band user edits can be detected. Git-independent.

`Checkpoint` (defined in `session.py`): `snapshots: list[FileSnapshot]` — one agent turn's edits (one snapshot per changed file), grouped as a single atomic undo/redo unit.

Key methods: `add_user_message`, `add_assistant_message`, `pop_last_message` (rollback), `clear_history` (clears message history only — a compaction hook for Week 5; leaves the undo/redo stacks intact).

---

### `llm_client.py` — LLM Client

Wraps `litellm.completion` with streaming and retry logic. Stateless — receives the full history on every call and returns a token iterator.

**Provider routing** is handled entirely by the litellm model string, set at startup:

| Provider  | litellm model string             |
|-----------|----------------------------------|
| Anthropic | `claude-3-5-sonnet-20241022`     |
| Gemini    | `gemini/gemini-2.5-flash`        |

**Retry policy:**

| Exception                  | Behaviour                                              |
|----------------------------|--------------------------------------------------------|
| `AuthenticationError`      | Raise `RuntimeError` immediately — no retry            |
| `RateLimitError` (daily)   | Raise `RuntimeError` immediately — retrying won't help |
| `RateLimitError` (transient) | Exponential backoff, max 3 attempts               |
| `APIConnectionError`       | Exponential backoff, max 3 attempts                    |

---

### `io_layer.py` — I/O Layer

Two independent classes with no shared state.

**`InputReader`** wraps `prompt_toolkit.PromptSession`:
- Persistent history backed by `~/.fizzy_history`
- `EOFError` on Ctrl-D → chat loop exits cleanly
- `KeyboardInterrupt` (Ctrl-C) → returns `""` → chat loop continues

**`OutputRenderer`** wraps `rich.Live`:
- Streams LLM output as `rich.Markdown`, re-rendering every ~20 characters to avoid CPU burn on large responses
- `finish_stream` performs a final full render and prints a blank separator line
- `print_error` / `print_info` for out-of-band messages

---

### `token_tracker.py` — Token Tracker

Read-only observer — never mutates session state. Uses `litellm.token_counter`, which is a **local calculation** (no API call) and works across all providers.

**Thresholds:**

| Status  | Condition         | Effect                            |
|---------|-------------------|-----------------------------------|
| `OK`    | < 80% of limit    | Silent                            |
| `WARN`  | ≥ 80% of limit    | Prints a yellow warning; proceeds |
| `BLOCK` | ≥ 95% of limit    | Prints an error; rolls back the user message; skips the LLM call |

On any counting error, falls back to `0` tokens (safely `OK`) rather than falsely blocking.

---

### `file_context.py` — File Context Manager

Stateless helper module. All state lives in `Session.context_files`; none of these functions touch `Session.history`.

| Function | Signature | Responsibility |
|---|---|---|
| `add_file` | `(session, path_str) → (bool, str)` | Resolve path (relative to `working_dir`), validate (exists, is file, UTF-8, ≤ 512 KB), no-op if already tracked, else store `FileEntry` |
| `drop_file` | `(session, path_str) → (bool, str)` | Remove entry from `context_files`; error if not tracked |
| `list_files` | `(session) → list[Path]` | Return tracked paths in insertion order |
| `refresh_files` | `(session) → list[str]` | Compare each file's current `st_mtime` to the stored value; re-read if changed, remove if deleted; return human-readable notices |
| `build_system_message` | `(session) → dict \| None` | Build `{"role": "system", "content": ...}` from all tracked files; returns `None` when no files are tracked |

**Constants:** `MAX_FILE_BYTES = 512 * 1024` (512 KB per file).

**System message** is constructed fresh before every LLM call by `chat_loop.py` and prepended to the messages list passed to `LLMClient.stream()`. It is never written into `Session.history`.

---

### `prompts.py` — Agent Instructions

Defines the static instruction string injected into the LLM system message whenever files are in context.

**`AGENT_INSTRUCTIONS: str`** — tells the LLM:
- The exact search/replace block format it must use for file edits
- That the SEARCH text must appear exactly once in the file
- That only files listed in the context may be edited
- That multiple blocks are allowed in a single response
- That non-edit replies should use plain text

This module is content-only — no functions, no state. `chat_loop.py` owns the decision of when to inject it and wraps it in `{"role": "system", "content": AGENT_INSTRUCTIONS}`.

---

### `edit_applier.py` — Edit Applier

Stateless module that parses LLM-proposed file edits and applies them with user confirmation.

**Search/replace block format** (LLM output):
```
<<<<<<< SEARCH src/foo.py
<text to find — must be unique in the file>
=======
<replacement text>
>>>>>>> REPLACE
```

| Symbol | Signature | Responsibility |
|---|---|---|
| `EditProposal` | dataclass: `path`, `search`, `replace` | Represents one validated edit extracted from the LLM response |
| `parse_proposals` | `(response, session) → list[EditProposal]` | Scan response for all blocks; resolve each filename against `working_dir`; skip files not in `session.context_files`; keep the structural trailing `\n` so `EditProposal.search` and `.replace` are fully line-terminated strings |
| `sanitize_for_display` | `(text) → str` | Wrap complete search/replace blocks in triple-backtick fences so Rich's Markdown renderer does not mangle the git-conflict-style markers; partial blocks (mid-stream) are left as-is |
| `compute_diff` | `(path, before, after) → str` | Unified diff of two whole-file contents via `difflib.unified_diff`; `""` when equal. Shared by the apply preview and `change_history`'s undo/redo preview |
| `try_replace` | `(content, search, replace) → ReplaceResult` | Shared primitive: locate `search` exactly once (0 → not-found, 2+ → ambiguous) with the narrow trailing-newline fallback, then substitute. Reports the new content plus the *effective* search/replace used |
| `apply_edits` | `(proposals, session, renderer, reader) → list[EditOutcome]` | Group proposals by file (first-seen order); fold each file's blocks over `session.context_files[path].content` into one combined whole-file change (a later block composing on an earlier one; an unlocatable block is reported and skipped). Files whose folded content is unchanged are no-ops. Show every changed file's diff, ask a **single atomic `[y/N]`** for the turn: on confirm write each changed file once; on decline write nothing. Returns one `EditOutcome` per file. Does **not** touch the undo/redo stacks |
| `EditOutcome` | dataclass: `path: Path`, `applied: bool`, `snapshot: FileSnapshot \| None` | Per-file result; `snapshot` (before/after/mtime) is set iff the file was written |

`apply_edits` writes each file exactly once per turn, so two same-turn edits to one file compose in memory (no intermediate write). The whole-file `before`/`after` snapshot is the unit of undo — there is no per-block fragment to re-locate, so the previous search/replace `ChangeRecord` / `_expand_to_unique` machinery is gone.

**Key invariant:** `session.context_files` and the undo/redo stacks are never updated by the applier. The next turn's `refresh_files()` call picks up the new content from disk, and `chat_loop` checkpoints the applied snapshots. Declined, search-not-found, ambiguous, and no-op edits never reach disk.

---

### `change_history.py` — Change History

Stateless helper module for git-independent undo/redo. All state lives in `Session.undo_stack` / `Session.redo_stack` (lists of `Checkpoint`); none of these functions touch `Session.history`. Follows the classic editor model: recording a new checkpoint clears the redo stack.

Undo/redo overwrite each file **wholesale** with the stored snapshot (`before` for undo, `after` for redo) — the file is the unit. There is no merge: an edit the user made outside fizzy is detected and the user is warned before it is overwritten.

**Divergence detection** answers "has the user touched this file since the tool last wrote it?" mtime is the fast path: if the on-disk mtime equals the path's last-write mtime the file is unchanged. If it differs, the on-disk content is compared against the state we expect (the `after` for undo, the `before` for redo) — a mismatch (or a missing file) means it diverged. All snapshots of a path share one last-write mtime, re-synced after every write (`_sync_mtime`), so undoing/redoing one checkpoint keeps divergence accurate for other checkpoints touching the same file.

| Function | Signature | Responsibility |
|---|---|---|
| `record_checkpoint` | `(session, snapshots) → Checkpoint \| None` | Group a turn's `FileSnapshot`s into one `Checkpoint`; push onto `undo_stack`; clear `redo_stack`; `None` if no snapshots |
| `undo_last` | `(session, renderer, reader) → Checkpoint \| None` | Revert the top checkpoint: write each file's `before`. **Atomic** — if every file is unchanged, apply directly with a brief message (no prompt); if any file diverged, show the diffs, warn, and ask **one** `[y/N]` (decline writes nothing). On success move the checkpoint `undo`→`redo` and re-sync mtimes; `None` on empty/decline |
| `redo_last` | `(session, renderer, reader) → Checkpoint \| None` | Symmetric to `undo_last`: write each file's `after`; move the checkpoint `redo`→`undo` |

Undo/redo are atomic over the whole checkpoint — all files or none. Both commands are invoked from `chat_loop.py`, which appends a one-line note to history on success.

---

### `chat_loop.py` — Chat Loop

The only component that calls more than one other component. Implements the main REPL turn:

```
1.  read_line()                    → blocks for user input
2.  strip + skip empty             → guard against whitespace-only input
3.  handle slash commands          → no LLM call for any branch
      /add <file>                  → file_context.add_file(); print result; continue
      /add (bad args)              → print usage hint; continue
      /drop <file>                 → file_context.drop_file(); print result; continue
      /drop (bad args)             → print usage hint; continue
      /files                       → file_context.list_files(); print relative paths
                                     (or "No files in context."); continue
      /undo                        → change_history.undo_last(); on success append a
                                     "Reverted the last change (files)." note to history; continue
      /redo                        → change_history.redo_last(); on success append a
                                     "Re-applied the last undone change (files)." note; continue
      /quit /exit /q               → exit
4.  add_user_message()             → append to session history
5.  refresh_files()                → re-read changed files; print notices
6.  build_system_message()         → assemble {role:system} from context files (or None)
    if files present:
      messages = [instructions_msg, system_msg] + history
                  └─ AGENT_INSTRUCTIONS  └─ file contents
    else:
      messages = history
7.  tracker.check(messages)        → BLOCK: rollback + continue; WARN: print + proceed
                                     (counts file content tokens too)
8.  client.stream(messages)        → iterate chunks inside renderer.start_stream() Live context
                                     each chunk passed to renderer as sanitize_for_display(accumulated)
                                     (wraps complete edit blocks in code fences; partial blocks unchanged)
9.  add_assistant_message()        → persist full response to session history
9.5 edit_applier.parse_proposals()  → extract edit blocks from reply
    if any proposals:
      edit_applier.apply_edits()    → group by file, fold each (composing),
                                      show diffs, ONE atomic confirm, write
                                      → list[EditOutcome] (before/after snapshots)
      change_history.record_checkpoint(applied snapshots)  → one checkpoint for the turn
      add_user_message(outcomes)   → e.g. "Edit to 'foo.py': applied." / "not applied — file unchanged."
                                     stored in history so LLM sees outcome on the next turn
10. rebuild final_messages with reply → [instructions_msg, system_msg] + history  or  history
    tracker.render_status()           → print token bar
```

**Key invariant:** neither system message is ever written into `session.history`. Both are assembled fresh on every turn and prepended only to the list passed to `client.stream()` and `tracker.check()`. When no files are in context, no system messages are prepended at all.

On any streaming error, the user message is rolled back via `session.pop_last_message()` so the history stays consistent and the loop continues.

---

### `main.py` — Entry Point

Bootstraps all components and wires them together. Owns:
- CLI argument parsing (`typer`)
- Provider → litellm model string mapping (`_PROVIDER_PREFIX`)
- Model → context window size table (`_MODEL_MAX_TOKENS`)
- Construction order: `Session` → `LLMClient` → `InputReader` + `OutputRenderer` → `TokenTracker` → `ChatLoop`

The model-to-token-limit table is intentionally hardcoded here for Week 1. It will be replaced by a dynamic registry in Week 5 (multi-model support).

---

## Data Flow

```
startup
  main.py reads CLI args / env vars
  constructs Session with model + max_tokens
  constructs all components
  calls ChatLoop.run()

per turn
  InputReader.read_line()
    → text

  Session.add_user_message(text)
    → history grows by 1

  TokenTracker.check(history)
    → litellm.token_counter() [local, no network]
    → TokenStatus + used count

  if BLOCK:
    Session.pop_last_message()
    OutputRenderer.print_error()
    → continue loop

  if WARN:
    OutputRenderer.print_info()

  LLMClient.stream(history)
    → litellm.completion(model, messages, api_key, stream=True)
    → yields text chunks

  OutputRenderer.append_chunk() [inside Live context]
    → rich.Live re-renders accumulated Markdown every ~20 chars

  OutputRenderer.finish_stream()
    → final render + blank line

  Session.add_assistant_message(accumulated)
    → history grows by 1

  TokenTracker.render_status()
    → prints coloured token bar
```

---

## Key Dependencies

| Library          | Role                                                        |
|------------------|-------------------------------------------------------------|
| `litellm`        | Unified LLM API — routes to Anthropic, Gemini, etc. Also provides local token counting via `token_counter` |
| `rich`           | Terminal rendering — Markdown, syntax highlighting, Live streaming display |
| `prompt_toolkit` | Line editing, input history, Ctrl-D / Ctrl-C handling       |
| `typer`          | CLI argument parsing with `--help` and env var support      |

---

## Provider Support

fizzy uses litellm's model string convention to select the provider. The `--provider` CLI flag maps a short provider name to the correct litellm prefix:

| `--provider` | litellm prefix | Example model string         | API key env var    |
|--------------|----------------|------------------------------|--------------------|
| `anthropic`  | _(none)_       | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY`|
| `gemini`     | `gemini/`      | `gemini/gemini-2.5-flash`    | `GEMINI_API_KEY`   |

Adding a new provider in future requires: adding an entry to `_PROVIDER_PREFIX`, `_PROVIDER_DEFAULT_MODEL`, `_PROVIDER_ENVVAR`, and `_MODEL_MAX_TOKENS` in `main.py`.

---

## Project Layout

```
fizzy/
├── ARCHITECTURE.md           # this document
├── REQUIREMENTS.md           # weekly milestones and feature plan
├── pyproject.toml            # dependencies and fizzy CLI entry point
├── .python-version           # pinned to 3.12
├── fizzy/
│   ├── main.py               # entry point, CLI, bootstrap
│   ├── session.py            # in-memory state (Session, FileEntry)
│   ├── llm_client.py         # litellm streaming client
│   ├── io_layer.py           # terminal input/output
│   ├── chat_loop.py          # REPL orchestration
│   ├── token_tracker.py      # token counting and budget enforcement
│   ├── file_context.py       # file context manager (Week 2)
│   ├── edit_applier.py       # search/replace edit applier (Week 2)
│   ├── change_history.py     # undo/redo change history (Week 2)
│   └── prompts.py            # static agent instruction strings (Week 2)
└── tests/
    ├── test_session.py
    ├── test_llm_client.py
    ├── test_io_layer.py
    ├── test_token_tracker.py
    ├── test_chat_loop.py
    ├── test_file_context.py  # (Week 2)
    ├── test_edit_applier.py  # (Week 2)
    ├── test_change_history.py # (Week 2)
    └── test_prompts.py       # (Week 2)
```
