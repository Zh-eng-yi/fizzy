"""Stateless helpers for managing files in the LLM context.

All state is stored on Session.context_files (dict[Path, FileEntry]).
None of these functions mutate Session.history.

Public API
----------
add_file      -- resolve, validate, and add a file to context
drop_file     -- remove a file from context
list_files    -- return ordered list of tracked paths
refresh_files -- re-read files whose mtime has changed since last read
build_system_message -- assemble the {"role": "system", ...} dict for the LLM
"""

from pathlib import Path

from fizzy.session import FileEntry, Session

MAX_FILE_BYTES = 512 * 1024  # 512 KB hard cap per file


def add_file(session: Session, path_str: str) -> tuple[bool, str]:
    """Resolve *path_str* relative to session.working_dir, validate it, and
    add it to session.context_files.

    Returns (True, success_message) or (False, error_message).
    """
    if not path_str:
        return False, "Error: no file path provided."

    path = (session.working_dir / path_str).resolve()

    if not path.exists():
        return False, f"Error: '{path_str}' not found."

    if not path.is_file():
        return False, f"Error: '{path_str}' is not a file."

    if path in session.context_files:
        return False, f"'{path_str}' is already in context."

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        limit_kb = MAX_FILE_BYTES // 1024
        actual_kb = size // 1024
        return (
            False,
            f"Error: '{path_str}' is {actual_kb} KB, which exceeds the {limit_kb} KB limit.",
        )

    # Read raw bytes once — used for both encoding check and content.
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return False, f"Error reading '{path_str}': {exc}"

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in raw:
            return (
                False,
                f"Error: '{path_str}' appears to be a binary file — only text files are supported.",
            )
        return (
            False,
            f"Error: '{path_str}' is not valid UTF-8 — only UTF-8 encoded text files are supported.",
        )

    # Guard against files that decoded but still contain null bytes (e.g. UTF-8
    # encoded U+0000), which strongly suggests binary content.
    if "\x00" in content:
        return (
            False,
            f"Error: '{path_str}' appears to be a binary file — only text files are supported.",
        )

    mtime = path.stat().st_mtime
    session.context_files[path] = FileEntry(content=content, mtime=mtime)
    return True, f"Added '{path_str}' to context."


def drop_file(session: Session, path_str: str) -> tuple[bool, str]:
    """Remove a tracked file from session.context_files.

    Returns (True, success_message) or (False, error_message).
    """
    path = (session.working_dir / path_str).resolve()
    if path not in session.context_files:
        return False, f"Error: '{path_str}' is not in context."
    del session.context_files[path]
    return True, f"Dropped '{path_str}' from context."


def list_files(session: Session) -> list[Path]:
    """Return the tracked paths in insertion order."""
    return list(session.context_files.keys())


def refresh_files(session: Session) -> list[str]:
    """Re-read any tracked file whose mtime differs from the stored value.

    Removes files that have been deleted from disk.
    Returns a list of human-readable change notices (empty when nothing changed).
    """
    notices: list[str] = []

    for path in list(session.context_files.keys()):
        if not path.exists():
            del session.context_files[path]
            notices.append(f"'{path.name}' removed from context (deleted on disk).")
            continue

        current_mtime = path.stat().st_mtime
        if current_mtime == session.context_files[path].mtime:
            continue  # unchanged — skip re-read

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            notices.append(f"Warning: could not re-read '{path.name}': {exc}")
            continue

        session.context_files[path] = FileEntry(content=content, mtime=current_mtime)
        notices.append(f"'{path.name}' reloaded (changed on disk).")

    return notices


def build_system_message(session: Session) -> dict | None:
    """Build the system message dict that injects all context-file contents.

    Returns None when no files are tracked (caller should skip injection).
    The returned dict is in the OpenAI/Anthropic message format:
        {"role": "system", "content": <str>}
    This message is constructed fresh on every LLM call and is NOT stored in
    Session.history.
    """
    if not session.context_files:
        return None

    parts: list[str] = ["The following files are part of the current context:\n"]
    for path, entry in session.context_files.items():
        parts.append(f"--- {path} ---\n{entry.content}")

    return {"role": "system", "content": "\n".join(parts)}
