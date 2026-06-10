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
apply_proposal   -- show diff, prompt y/N, write to disk if confirmed
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from fizzy.io_layer import InputReader, OutputRenderer
from fizzy.session import Session

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


def apply_proposal(
    proposal: EditProposal,
    session: Session,
    renderer: OutputRenderer,
    reader: InputReader,
) -> bool:
    """Display a diff, prompt the user for confirmation, and apply if approved.

    Returns True if the edit was written to disk, False otherwise.

    The session.context_files entry for the file is intentionally *not*
    updated here — the next turn's refresh_files() call will pick up the
    new mtime and content from disk.
    """
    content = session.context_files[proposal.path].content
    search = proposal.search
    replace = proposal.replace
    count = content.count(search)

    # Narrow fallback: parse_proposals always appends a structural \n, but a
    # file's very last line may have no trailing newline.  Try matching without
    # the trailing \n on both search and replace so we don't accidentally add a
    # newline that was never there.
    if count == 0 and search.endswith("\n") and not content.endswith("\n"):
        stripped_search = search[:-1]
        stripped_replace = replace[:-1] if replace.endswith("\n") else replace
        fallback_count = content.count(stripped_search)
        if fallback_count >= 1:
            count = fallback_count
            search = stripped_search
            replace = stripped_replace

    if count == 0:
        renderer.print_error(
            f"Edit not applied: search text not found in '{proposal.path.name}'."
        )
        return False

    if count > 1:
        renderer.print_error(
            f"Edit not applied: search text is ambiguous — found {count} times "
            f"in '{proposal.path.name}'. "
            "Provide a larger block that uniquely identifies the target."
        )
        return False

    new_content = content.replace(search, replace, 1)

    diff = compute_diff(proposal)
    renderer.print_info(diff)

    response = reader.read_line(f"Apply this change to '{proposal.path.name}'? [y/N]: ")
    if response.strip().lower() in ("y", "yes"):
        proposal.path.write_text(new_content, encoding="utf-8")
        return True

    return False
