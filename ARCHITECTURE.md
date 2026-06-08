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
```

`FileEntry` (defined in `session.py`): `content: str`, `mtime: float` — the file's text and the `os.stat().st_mtime` at last read.

Key methods: `add_user_message`, `add_assistant_message`, `pop_last_message` (rollback), `clear_history` (compaction hook for Week 5).

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
      /quit /exit /q               → exit
4.  add_user_message()             → append to session history
5.  refresh_files()                → re-read changed files; print notices
6.  build_system_message()         → assemble {role:system} from context files (or None)
    messages = [system_msg] + history  if files present  else  history
7.  tracker.check(messages)        → BLOCK: rollback + continue; WARN: print + proceed
                                     (counts file content tokens too)
8.  client.stream(messages)        → iterate chunks inside renderer.start_stream() Live context
9.  add_assistant_message()        → persist full response to session history
10. rebuild messages with reply    → [system_msg] + history  or  history
    tracker.render_status()        → print token bar
```

**Key invariant:** the system message is never written into `session.history`. It is built fresh from `session.context_files` on every turn and prepended only to the list passed to `client.stream()` and `tracker.check()`.

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
│   └── file_context.py       # file context manager (Week 2)
└── tests/
    ├── test_session.py
    ├── test_llm_client.py
    ├── test_io_layer.py
    ├── test_token_tracker.py
    ├── test_chat_loop.py
    └── test_file_context.py  # (Week 2)
```
