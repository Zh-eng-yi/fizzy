"""Parse and apply LLM-proposed file edits.

The LLM is expected to signal edits using search/replace blocks:

    <<<<<<< SEARCH src/foo.py
    <text to find — must be unique within the file>
    =======
    <replacement text>
    >>>>>>> REPLACE

A single response may contain multiple blocks for different files.
Only blocks whose target file is already in session.context_files are
processed; others are silently skipped.

Public API
----------
parse_proposals  -- extract all valid EditProposals from an LLM response
compute_diff     -- unified diff of two whole-file contents, for display
try_replace      -- locate a unique search and substitute (shared primitive)
apply_edits      -- group proposals by file, fold each, one atomic y/N, write
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.session import FileSnapshot, Session

# Matches exactly the 7-character git-conflict-style markers.
_PATTERN = re.compile(
    r"<<<<<<< SEARCH (.+?)\n"   # group 1: filename (trimmed below)
    r"(.*?)"                    # group 2: search text
    r"^=======\n"               # separator (must start a line)
    r"(.*?)"                    # group 3: replace text
    r"^>>>>>>> REPLACE",        # end marker (must start a line)
    re.DOTALL | re.MULTILINE,
)


@dataclass
class EditProposal:
    """A single search/replace edit proposed by the LLM."""

    path: Path    # resolved absolute path of the target file
    search: str   # text to locate — must appear exactly once in the file
    replace: str  # text to substitute in


class ReplaceResult(NamedTuple):
    """Outcome of try_replace.

    On success ``content`` is the new text and ``error`` is empty; ``search``
    and ``replace`` are the *effective* fragments used (possibly stripped of a
    trailing newline by the narrow fallback).  On conflict ``content`` is None
    and ``error`` explains why (text not found, or ambiguous).
    """

    content: str | None
    search: str
    replace: str
    error: str


def parse_proposals(response: str, session: Session) -> list[EditProposal]:
    """Extract all well-formed search/replace blocks from *response*.

    Only proposals whose resolved path is present in session.context_files
    are returned.  Blocks for unknown files are silently dropped.
    """
    proposals: list[EditProposal] = []
    for match in _PATTERN.finditer(response):
        filename = match.group(1).strip()
        # The regex captures the newline immediately before ======= (group 2) and
        # before >>>>>>> REPLACE (group 3) as part of the match.  We keep those
        # newlines: EditProposal.search and .replace are fully line-terminated
        # strings that match real file content precisely.
        search = match.group(2)
        replace = match.group(3)
        path = (session.working_dir / filename).resolve()
        if path in session.context_files:
            proposals.append(EditProposal(path=path, search=search, replace=replace))
    return proposals


def sanitize_for_display(text: str) -> str:
    """Wrap complete search/replace blocks in triple-backtick code fences.

    Rich's Markdown renderer mangles the git-conflict-style markers:
    - ``<<<<<<<`` is treated as an HTML opening tag
    - ``=======`` is treated as a setext H2 heading underline (bold heading)
    - ``>>>>>>>`` is treated as a 7-level blockquote (rendered as ``| | | | | | |``)

    Wrapping complete blocks in code fences forces preformatted rendering.
    Partial blocks (stream not yet complete, no end marker present) are left
    as-is and may render oddly until the stream finishes.
    """
    return _PATTERN.sub(lambda m: f"```\n{m.group(0)}\n```", text)


def compute_diff(path: Path, before: str, after: str) -> str:
    """Return a unified diff between two whole-file contents.

    Uses *path* as the diff header so the output clearly names the affected
    file.  Returns an empty string when *before* == *after*.  Shared by the edit
    applier (apply preview) and change_history (undo/redo preview).

    Lines are split *without* keepends and emitted with ``lineterm=""``, so each
    diff line is newline-free and joined with ``\\n``.  This keeps a file whose
    last line lacks a trailing newline from gluing its ``-`` and ``+`` lines onto
    the same rendered line.
    """
    name = str(path)
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=name,
            tofile=name,
            lineterm="",
        )
    )


def try_replace(content: str, search: str, replace: str) -> ReplaceResult:
    """Locate ``search`` exactly once in ``content`` and substitute ``replace``.

    Includes the narrow trailing-newline fallback: when ``search`` ends in a
    newline that the file's final line lacks, both fragments are retried without
    that trailing newline so a last-line edit matches without inserting a
    spurious newline.

    The reported ``search``/``replace`` are the *effective* fragments used.
    """
    count = content.count(search)

    if count == 0 and search.endswith("\n") and not content.endswith("\n"):
        stripped_search = search[:-1]
        stripped_replace = replace[:-1] if replace.endswith("\n") else replace
        if content.count(stripped_search) >= 1:
            search = stripped_search
            replace = stripped_replace
            count = content.count(search)

    if count == 0:
        return ReplaceResult(None, search, replace, "search text not found")
    if count > 1:
        return ReplaceResult(
            None, search, replace, f"search text is ambiguous — found {count} times"
        )

    new_content = content.replace(search, replace, 1)
    return ReplaceResult(new_content, search, replace, "")


@dataclass
class EditOutcome:
    """Result of applying one file's edits in a turn.

    ``applied`` is True iff the file was written.  ``snapshot`` is the whole-file
    before/after FileSnapshot for the change history (set iff applied; None for a
    declined or no-op file).  The chat loop reports outcomes to the LLM and
    checkpoints the applied snapshots.
    """

    path: Path
    applied: bool
    snapshot: FileSnapshot | None


def _fold_file(
    path: Path, before: str, proposals: list[EditProposal], renderer: OutputRenderer
) -> str:
    """Apply every block for one file to an in-memory copy and return the result.

    Blocks are applied in order, each composing on the previous one's output.  A
    block whose search text cannot be uniquely located is reported and skipped.
    Returns the folded content (== *before* when nothing applied).
    """
    working = before
    for proposal in proposals:
        result = try_replace(working, proposal.search, proposal.replace)
        if result.content is None:
            message = f"Edit not applied to '{path.name}': {result.error}."
            if "ambiguous" in result.error:
                message += " Provide a larger block that uniquely identifies the target."
            renderer.print_error(message)
            continue
        working = result.content
    return working


def apply_edits(
    proposals: list[EditProposal],
    session: Session,
    renderer: OutputRenderer,
    reader: InputReader,
) -> list[EditOutcome]:
    """Apply a turn's edit proposals, grouped per file, atomically.

    Proposals are grouped by target file (first-seen order).  Each file's blocks
    are folded over the once-per-turn snapshot in ``session.context_files`` into
    one combined whole-file change (a later block composing on an earlier one).
    Files whose folded content is unchanged (e.g. every block conflicted) are
    no-ops.  The diffs for all changed files are shown, then a SINGLE ``[y/N]``
    is asked for the whole turn: on confirm every changed file is written once;
    on decline nothing is written.

    Returns one EditOutcome per file (in first-seen order) carrying the
    before/after FileSnapshot for applied files.  The undo/redo stacks and
    session.context_files are intentionally not touched here: the chat loop
    checkpoints the snapshots, and the next refresh_files() re-reads from disk.
    """
    grouped: dict[Path, list[EditProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.path, []).append(proposal)

    # Fold each file in memory; split into changed vs. no-op.
    changed: list[tuple[Path, str, str]] = []   # (path, before, after)
    noops: list[Path] = []
    for path, file_proposals in grouped.items():
        before = session.context_files[path].content
        after = _fold_file(path, before, file_proposals, renderer)
        if after == before:
            noops.append(path)
        else:
            changed.append((path, before, after))

    applied_paths: set[Path] = set()
    if changed:
        for path, before, after in changed:
            renderer.print_info(compute_diff(path, before, after))
        names = ", ".join(sorted({path.name for path, _, _ in changed}))
        response = reader.read_line(
            f"Apply {len(changed)} change(s) to {names}? [y/N]: "
        )
        if response.strip().lower() in ("y", "yes"):
            for path, _, after in changed:
                path.write_text(after, encoding="utf-8")
                applied_paths.add(path)

    # Build outcomes in first-seen file order.
    outcomes: list[EditOutcome] = []
    after_by_path = {path: (before, after) for path, before, after in changed}
    for path in grouped:
        if path in applied_paths:
            before, after = after_by_path[path]
            snapshot = FileSnapshot(
                path=path,
                before=before,
                after=after,
                mtime=path.stat().st_mtime,
            )
            outcomes.append(EditOutcome(path=path, applied=True, snapshot=snapshot))
        else:
            outcomes.append(EditOutcome(path=path, applied=False, snapshot=None))
    return outcomes
