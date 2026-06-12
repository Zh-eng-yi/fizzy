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
compute_diff     -- produce a unified diff string for display
try_replace      -- locate a unique search and substitute (shared primitive)
apply_proposal   -- show diff, prompt y/N, write to disk if confirmed
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.session import ChangeRecord, Session

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


def compute_diff(proposal: EditProposal) -> str:
    """Return a unified diff string between the search text and the replace text.

    Uses the proposal's filename as the diff header so the output clearly
    names the affected file.  Returns an empty string when search == replace.
    """
    # parse_proposals always produces \n-terminated search and replace strings,
    # so splitlines(keepends=True) yields properly terminated lines and
    # unified_diff emits each change on its own line.
    old_lines = proposal.search.splitlines(keepends=True)
    new_lines = proposal.replace.splitlines(keepends=True)
    filename = str(proposal.path)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=filename,
            tofile=filename,
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


def _expand_to_unique(
    content: str, new_content: str, search: str, replace: str
) -> tuple[str, str]:
    """Widen (search, replace) with surrounding context until ``replace`` is
    uniquely locatable in ``new_content``.

    ``search`` occurs exactly once in ``content`` and
    ``new_content == content.replace(search, replace, 1)``.  Context lines are
    identical on both sides (unchanged text), so the widened pair remains a
    valid inverse edit.  Converges at worst to the whole file.
    """
    # Already uniquely undoable → record the fragment verbatim.
    if new_content.count(replace) == 1:
        return search, replace

    idx = content.index(search)        # unique match → single location
    left = idx                         # shared (unchanged) prefix boundary
    right_old = idx + len(search)      # suffix boundary in content
    right_new = idx + len(replace)     # suffix boundary in new_content

    while True:
        rec_search = content[left:right_old]
        rec_replace = new_content[left:right_new]
        if new_content.count(rec_replace) == 1:
            return rec_search, rec_replace

        # Prefer extending forward by a whole line, then backward by a line.
        if right_old < len(content):
            nl = content.find("\n", right_old)
            new_right = len(content) if nl == -1 else nl + 1
            right_new += new_right - right_old  # identical suffix on both sides
            right_old = new_right
        elif left > 0:
            nl = content.rfind("\n", 0, left - 1)
            left = 0 if nl == -1 else nl + 1
        else:
            # Whole file consumed: the entire new_content is trivially unique.
            return content[left:right_old], new_content[left:right_new]


def apply_proposal(
    proposal: EditProposal,
    session: Session,
    renderer: OutputRenderer,
    reader: InputReader,
) -> ChangeRecord | None:
    """Display a diff, prompt for confirmation, and apply the edit if approved.

    Returns the applied edit as a ChangeRecord (the inverse-applicable, possibly
    context-widened fragment) on success, or None if the search text could not
    be uniquely located or the user declined.

    The undo/redo stacks and session.context_files are intentionally not touched
    here: the chat loop groups returned records into a checkpoint, and the next
    refresh_files() picks up the new content from disk.
    """
    content = session.context_files[proposal.path].content
    result = try_replace(content, proposal.search, proposal.replace)

    if result.content is None:
        message = f"Edit not applied to '{proposal.path.name}': {result.error}."
        if "ambiguous" in result.error:
            message += " Provide a larger block that uniquely identifies the target."
        renderer.print_error(message)
        return None

    diff = compute_diff(proposal)
    renderer.print_info(diff)

    response = reader.read_line(f"Apply this change to '{proposal.path.name}'? [y/N]: ")
    if response.strip().lower() not in ("y", "yes"):
        return None

    proposal.path.write_text(result.content, encoding="utf-8")
    rec_search, rec_replace = _expand_to_unique(
        content, result.content, result.search, result.replace
    )
    return ChangeRecord(path=proposal.path, search=rec_search, replace=rec_replace)
