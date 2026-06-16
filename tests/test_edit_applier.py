from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fizzy.edit_applier import (
    EditOutcome,
    EditProposal,
    apply_edits,
    compute_diff,
    parse_proposals,
    sanitize_for_display,
    try_replace,
)
from fizzy.session import FileEntry, FileSnapshot, Session


# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def session(tmp_path):
    return Session(model="test-model", working_dir=tmp_path, max_tokens=100_000)


def _block(filename: str, search: str, replace: str) -> str:
    """Build a single well-formed search/replace block."""
    return (
        f"<<<<<<< SEARCH {filename}\n"
        f"{search}"
        f"=======\n"
        f"{replace}"
        f">>>>>>> REPLACE"
    )


def _mock_renderer():
    return MagicMock()


def _mock_reader(response: str = "y"):
    reader = MagicMock()
    reader.read_line.return_value = response
    return reader


def _seed(session: Session, tmp_path: Path, name: str, content: str) -> Path:
    """Write a file, add it to session.context_files, return its resolved path."""
    path = (tmp_path / name)
    path.write_text(content, encoding="utf-8")
    resolved = path.resolve()
    session.context_files[resolved] = FileEntry(content=content, mtime=path.stat().st_mtime)
    return resolved


# ── parse_proposals ───────────────────────────────────────────────────────────


class TestParseProposals:
    def test_empty_response_returns_empty_list(self, session):
        assert parse_proposals("No edits here.", session) == []

    def test_single_valid_block_returns_one_proposal(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old\n")
        proposals = parse_proposals(_block("foo.py", "old\n", "new\n"), session)
        assert len(proposals) == 1

    def test_proposal_has_correct_resolved_path(self, session, tmp_path):
        resolved = _seed(session, tmp_path, "foo.py", "old\n")
        proposals = parse_proposals(_block("foo.py", "old\n", "new\n"), session)
        assert proposals[0].path == resolved

    def test_proposal_has_correct_search_text(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old content\n")
        proposals = parse_proposals(_block("foo.py", "old content\n", "new\n"), session)
        # parse_proposals preserves the captured \n — it is part of the content
        assert proposals[0].search == "old content\n"

    def test_proposal_has_correct_replace_text(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old\n")
        proposals = parse_proposals(_block("foo.py", "old\n", "new content\n"), session)
        # parse_proposals preserves the captured \n — it is part of the content
        assert proposals[0].replace == "new content\n"

    def test_path_resolved_relative_to_working_dir(self, session, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        resolved = _seed(session, subdir, "bar.py", "x\n")
        # path given to parser is relative: "src/bar.py"
        proposals = parse_proposals(_block("src/bar.py", "x\n", "y\n"), session)
        assert proposals[0].path == resolved

    def test_file_not_in_context_is_skipped(self, session):
        proposals = parse_proposals(_block("ghost.py", "old\n", "new\n"), session)
        assert proposals == []

    def test_multiple_valid_blocks_returns_all(self, session, tmp_path):
        _seed(session, tmp_path, "a.py", "old\n")
        _seed(session, tmp_path, "b.py", "old\n")
        response = _block("a.py", "old\n", "new_a\n") + "\n\n" + _block("b.py", "old\n", "new_b\n")
        proposals = parse_proposals(response, session)
        assert len(proposals) == 2

    def test_mixed_valid_and_invalid_blocks(self, session, tmp_path):
        resolved = _seed(session, tmp_path, "real.py", "old\n")
        response = _block("real.py", "old\n", "new\n") + "\n\n" + _block("ghost.py", "old\n", "new\n")
        proposals = parse_proposals(response, session)
        assert len(proposals) == 1
        assert proposals[0].path == resolved

    def test_block_embedded_in_prose_is_found(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old\n")
        response = (
            "Here is the fix:\n\n"
            + _block("foo.py", "old\n", "new\n")
            + "\n\nLet me know if that looks good."
        )
        proposals = parse_proposals(response, session)
        assert len(proposals) == 1

    def test_malformed_block_missing_separator_ignored(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old\n")
        response = "<<<<<<< SEARCH foo.py\nold\n>>>>>>> REPLACE"
        assert parse_proposals(response, session) == []

    def test_malformed_block_missing_end_marker_ignored(self, session, tmp_path):
        _seed(session, tmp_path, "foo.py", "old\n")
        response = "<<<<<<< SEARCH foo.py\nold\n=======\nnew\n"
        assert parse_proposals(response, session) == []

    def test_search_text_preserves_trailing_newline(self, session, tmp_path):
        # The regex captures the structural \n before ======= as part of group 2.
        # parse_proposals keeps it so that EditProposal.search is a complete,
        # line-terminated string — matching real file content precisely.
        _seed(session, tmp_path, "foo.py", "old\n")
        proposals = parse_proposals(_block("foo.py", "old\n", "new\n"), session)
        assert proposals[0].search.endswith("\n")

    def test_replace_text_preserves_trailing_newline(self, session, tmp_path):
        # Same design applies to group 3 (the \n before >>>>>>>).
        _seed(session, tmp_path, "foo.py", "old\n")
        proposals = parse_proposals(_block("foo.py", "old\n", "new\n"), session)
        assert proposals[0].replace.endswith("\n")


# ── compute_diff ──────────────────────────────────────────────────────────────


class TestComputeDiff:
    """compute_diff now takes whole-file before/after contents and returns a
    unified diff, used for display before confirmation and during undo/redo."""

    def _path(self, tmp_path: Path) -> Path:
        return (tmp_path / "foo.py").resolve()

    def test_returns_string(self, tmp_path):
        assert isinstance(compute_diff(self._path(tmp_path), "a\n", "b\n"), str)

    def test_identical_content_is_empty(self, tmp_path):
        assert compute_diff(self._path(tmp_path), "same\n", "same\n") == ""

    def test_removed_line_has_minus_prefix(self, tmp_path):
        diff = compute_diff(self._path(tmp_path), "old line\n", "new line\n")
        assert any(l.startswith("-") and "old line" in l for l in diff.splitlines())

    def test_added_line_has_plus_prefix(self, tmp_path):
        diff = compute_diff(self._path(tmp_path), "old line\n", "new line\n")
        assert any(l.startswith("+") and "new line" in l for l in diff.splitlines())

    def test_contains_filename(self, tmp_path):
        diff = compute_diff(self._path(tmp_path), "a\n", "b\n")
        assert "foo.py" in diff

    def test_preserves_unchanged_context_lines(self, tmp_path):
        diff = compute_diff(self._path(tmp_path), "keep\nold\n", "keep\nnew\n")
        # the unchanged "keep" line appears as context (space-prefixed)
        assert any(l.startswith(" ") and "keep" in l for l in diff.splitlines())

    def test_changed_lines_appear_on_separate_lines(self, tmp_path):
        diff = compute_diff(self._path(tmp_path), "old line\n", "new line\n")
        minus = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(minus) == 1
        assert len(plus) == 1

    def test_last_line_without_trailing_newline_separates_changes(self, tmp_path):
        # Regression: whole-file contents whose last line lacks a trailing newline
        # must not glue the '-' and '+' lines onto the same rendered line.
        diff = compute_diff(self._path(tmp_path), "print('hi')", "print('bye')")
        minus = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(minus) == 1
        assert len(plus) == 1
        assert "print('hi')" in minus[0]
        assert "print('bye')" in plus[0]


# ── try_replace ───────────────────────────────────────────────────────────────


class TestTryReplace:
    """The shared search→replace primitive: locate the search text exactly once
    (with the narrow trailing-newline fallback) and substitute, reporting the
    effective texts used or a conflict reason."""

    def test_unique_match_returns_new_content(self):
        result = try_replace("a\nold\nb\n", "old\n", "new\n")
        assert result.content == "a\nnew\nb\n"
        assert result.error == ""

    def test_unique_match_reports_effective_search_and_replace(self):
        result = try_replace("a\nold\nb\n", "old\n", "new\n")
        assert result.search == "old\n"
        assert result.replace == "new\n"

    def test_replaces_only_first_occurrence_region(self):
        result = try_replace("pre\nTARGET\npost\n", "TARGET\n", "DONE\n")
        assert result.content == "pre\nDONE\npost\n"

    def test_not_found_returns_none_content(self):
        result = try_replace("a\nb\n", "missing\n", "new\n")
        assert result.content is None

    def test_not_found_reports_error(self):
        result = try_replace("a\nb\n", "missing\n", "new\n")
        assert "not found" in result.error.lower()

    def test_ambiguous_returns_none_content(self):
        result = try_replace("dup\ndup\n", "dup\n", "x\n")
        assert result.content is None

    def test_ambiguous_reports_error(self):
        result = try_replace("dup\ndup\n", "dup\n", "x\n")
        assert "ambiguous" in result.error.lower()

    def test_narrow_fallback_matches_file_without_trailing_newline(self):
        result = try_replace("print('x')", "print('x')\n", "print('y')\n")
        assert result.content == "print('y')"

    def test_narrow_fallback_reports_stripped_effective_texts(self):
        result = try_replace("print('x')", "print('x')\n", "print('y')\n")
        assert result.search == "print('x')"
        assert result.replace == "print('y')"


# ── sanitize_for_display ──────────────────────────────────────────────────────


class TestSanitizeForDisplay:
    """The raw LLM response is passed through sanitize_for_display before being
    handed to the Markdown renderer.  Complete edit blocks must be wrapped in
    triple-backtick fences so that Rich does not mangle the git-conflict-style
    markers (<<<<<<< is treated as HTML, ======= as a setext heading, and
    >>>>>>> as a deep blockquote).
    """

    def test_non_block_text_unchanged(self):
        text = "Here is some plain explanation.\n"
        assert sanitize_for_display(text) == text

    def test_empty_string_unchanged(self):
        assert sanitize_for_display("") == ""

    def test_complete_block_wrapped_in_code_fences(self):
        result = sanitize_for_display(_block("foo.py", "old\n", "new\n"))
        assert "```" in result

    def test_wrapped_block_starts_with_fence(self):
        result = sanitize_for_display(_block("foo.py", "old\n", "new\n"))
        assert result.startswith("```")

    def test_wrapped_block_preserves_all_markers(self):
        result = sanitize_for_display(_block("foo.py", "old\n", "new\n"))
        assert "<<<<<<< SEARCH foo.py" in result
        assert "=======" in result
        assert ">>>>>>> REPLACE" in result

    def test_wrapped_block_preserves_search_and_replace_text(self):
        result = sanitize_for_display(_block("foo.py", "old content\n", "new content\n"))
        assert "old content" in result
        assert "new content" in result

    def test_partial_block_missing_end_marker_unchanged(self):
        # Without >>>>>>> REPLACE the pattern should not match.
        partial = "<<<<<<< SEARCH foo.py\nold\n=======\nnew\n"
        assert sanitize_for_display(partial) == partial

    def test_partial_block_missing_separator_unchanged(self):
        # Without ======= the pattern should not match.
        partial = "<<<<<<< SEARCH foo.py\nold\n>>>>>>> REPLACE"
        assert sanitize_for_display(partial) == partial

    def test_multiple_blocks_each_wrapped(self):
        text = (
            _block("a.py", "old_a\n", "new_a\n")
            + "\n\n"
            + _block("b.py", "old_b\n", "new_b\n")
        )
        result = sanitize_for_display(text)
        # Two opening fences + two closing fences = 4 occurrences of ```
        assert result.count("```") == 4

    def test_block_in_prose_prose_is_preserved(self):
        text = (
            "Here is the fix:\n\n"
            + _block("foo.py", "old\n", "new\n")
            + "\n\nLet me know if that looks good."
        )
        result = sanitize_for_display(text)
        assert "Here is the fix:" in result
        assert "Let me know if that looks good." in result


# ── apply_edits ───────────────────────────────────────────────────────────────


class TestApplyEdits:
    """apply_edits groups proposals by file, folds each file's blocks into one
    combined whole-file change, shows the diffs, asks a SINGLE atomic [y/N] for
    the whole turn, and writes confirmed files once each.  It returns one
    EditOutcome per file carrying the before/after FileSnapshot for the change
    history.  Snapshots replace the old per-block ChangeRecord model.
    """

    def _setup(self, tmp_path: Path, files: dict[str, str]):
        session = Session(model="test", working_dir=tmp_path, max_tokens=100_000)
        paths = {name: _seed(session, tmp_path, name, content) for name, content in files.items()}
        return session, paths

    # single file

    def test_single_file_applies_and_writes(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert paths["foo.py"].read_text() == "new\n"
        assert len(outcomes) == 1
        assert outcomes[0].applied is True

    def test_outcome_is_editoutcome_with_path(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert isinstance(outcomes[0], EditOutcome)
        assert outcomes[0].path == paths["foo.py"]

    def test_snapshot_holds_before_and_after(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        snap = outcomes[0].snapshot
        assert isinstance(snap, FileSnapshot)
        assert snap.before == "old\n"
        assert snap.after == "new\n"

    def test_snapshot_records_write_mtime(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert outcomes[0].snapshot.mtime == paths["foo.py"].stat().st_mtime

    # grouping / composition — one snapshot per file

    def test_two_blocks_same_file_compose_into_one_snapshot(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "a\nb\nc\n"})
        outcomes = apply_edits(
            [
                EditProposal(path=paths["foo.py"], search="a\n", replace="A\n"),
                EditProposal(path=paths["foo.py"], search="c\n", replace="C\n"),
            ],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert paths["foo.py"].read_text() == "A\nb\nC\n"
        assert len(outcomes) == 1                       # grouped per file
        assert outcomes[0].snapshot.before == "a\nb\nc\n"
        assert outcomes[0].snapshot.after == "A\nb\nC\n"

    def test_later_block_sees_earlier_block_result(self, tmp_path):
        # block 2's search ("y") only exists AFTER block 1 (x→y) applies.
        session, paths = self._setup(tmp_path, {"foo.py": "x\n"})
        apply_edits(
            [
                EditProposal(path=paths["foo.py"], search="x\n", replace="y\n"),
                EditProposal(path=paths["foo.py"], search="y\n", replace="z\n"),
            ],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert paths["foo.py"].read_text() == "z\n"

    # multiple files — atomic single confirmation

    def test_two_files_two_outcomes_both_written(self, tmp_path):
        session, paths = self._setup(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
        outcomes = apply_edits(
            [
                EditProposal(path=paths["a.py"], search="a\n", replace="A\n"),
                EditProposal(path=paths["b.py"], search="b\n", replace="B\n"),
            ],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert paths["a.py"].read_text() == "A\n"
        assert paths["b.py"].read_text() == "B\n"
        assert len(outcomes) == 2
        assert all(o.applied for o in outcomes)

    def test_single_confirmation_for_all_files(self, tmp_path):
        session, paths = self._setup(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
        reader = _mock_reader("y")
        apply_edits(
            [
                EditProposal(path=paths["a.py"], search="a\n", replace="A\n"),
                EditProposal(path=paths["b.py"], search="b\n", replace="B\n"),
            ],
            session, _mock_renderer(), reader,
        )
        assert reader.read_line.call_count == 1

    def test_decline_writes_nothing_atomically(self, tmp_path):
        session, paths = self._setup(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
        outcomes = apply_edits(
            [
                EditProposal(path=paths["a.py"], search="a\n", replace="A\n"),
                EditProposal(path=paths["b.py"], search="b\n", replace="B\n"),
            ],
            session, _mock_renderer(), _mock_reader("n"),
        )
        assert paths["a.py"].read_text() == "a\n"
        assert paths["b.py"].read_text() == "b\n"
        assert all(not o.applied for o in outcomes)
        assert all(o.snapshot is None for o in outcomes)

    # diff display + prompt

    def test_diff_shown_before_confirmation(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        renderer = _mock_renderer()
        apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, renderer, _mock_reader("y"),
        )
        renderer.print_info.assert_called()

    def test_prompt_mentions_filename(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        reader = _mock_reader("y")
        apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), reader,
        )
        prompt = reader.read_line.call_args.args[0]
        assert "foo.py" in prompt

    # conflicting blocks are skipped, others still apply

    def test_conflicting_block_skipped_others_apply(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "a\nb\nc\n"})
        renderer = _mock_renderer()
        outcomes = apply_edits(
            [
                EditProposal(path=paths["foo.py"], search="a\n", replace="A\n"),
                EditProposal(path=paths["foo.py"], search="missing\n", replace="Z\n"),
            ],
            session, renderer, _mock_reader("y"),
        )
        assert paths["foo.py"].read_text() == "A\nb\nc\n"
        assert outcomes[0].applied is True
        renderer.print_error.assert_called()

    # no-op: a file whose only block fails to match is not a change

    def test_unmatched_only_block_is_noop_not_applied(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "real\n"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="missing\n", replace="Z\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert paths["foo.py"].read_text() == "real\n"
        assert outcomes[0].applied is False
        assert outcomes[0].snapshot is None

    def test_no_changes_does_not_prompt(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "real\n"})
        reader = _mock_reader("y")
        apply_edits(
            [EditProposal(path=paths["foo.py"], search="missing\n", replace="Z\n")],
            session, _mock_renderer(), reader,
        )
        reader.read_line.assert_not_called()

    # narrow trailing-newline fallback (last line, no newline)

    def test_file_without_trailing_newline_is_matched(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "print('hi')"})
        outcomes = apply_edits(
            [EditProposal(path=paths["foo.py"], search="print('hi')\n", replace="print('bye')\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert outcomes[0].applied is True
        assert paths["foo.py"].read_text() == "print('bye')"   # no trailing \n added

    # isolation

    def test_context_files_not_updated(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        # refresh_files() picks this up next turn — not apply_edits' job
        assert session.context_files[paths["foo.py"]].content == "old\n"

    def test_does_not_touch_undo_stack(self, tmp_path):
        session, paths = self._setup(tmp_path, {"foo.py": "old\n"})
        apply_edits(
            [EditProposal(path=paths["foo.py"], search="old\n", replace="new\n")],
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert session.undo_stack == []

    def test_empty_proposals_returns_empty(self, tmp_path):
        session, _ = self._setup(tmp_path, {})
        assert apply_edits([], session, _mock_renderer(), _mock_reader("y")) == []
