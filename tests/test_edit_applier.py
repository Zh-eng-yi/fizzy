from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fizzy.edit_applier import EditProposal, apply_proposal, compute_diff, parse_proposals, sanitize_for_display
from fizzy.session import FileEntry, Session


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
    def _proposal(self, tmp_path: Path, search: str, replace: str) -> EditProposal:
        return EditProposal(path=(tmp_path / "foo.py").resolve(), search=search, replace=replace)

    def test_returns_string(self, tmp_path):
        assert isinstance(compute_diff(self._proposal(tmp_path, "old\n", "new\n")), str)

    def test_removed_line_has_minus_prefix(self, tmp_path):
        diff = compute_diff(self._proposal(tmp_path, "old line\n", "new line\n"))
        assert any(line.startswith("-") and "old line" in line for line in diff.splitlines())

    def test_added_line_has_plus_prefix(self, tmp_path):
        diff = compute_diff(self._proposal(tmp_path, "old line\n", "new line\n"))
        assert any(line.startswith("+") and "new line" in line for line in diff.splitlines())

    def test_diff_contains_filename(self, tmp_path):
        diff = compute_diff(self._proposal(tmp_path, "old\n", "new\n"))
        assert "foo.py" in diff

    def test_changed_lines_appear_on_separate_lines(self, tmp_path):
        # parse_proposals preserves the structural \n, so compute_diff always
        # receives properly newline-terminated strings.  Each changed line must
        # appear on its own diff line (not concatenated with the next).
        diff = compute_diff(self._proposal(tmp_path, "old line\n", "new line\n"))
        minus = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus  = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(minus) == 1
        assert len(plus) == 1

    def test_unchanged_content_produces_empty_diff(self, tmp_path):
        diff = compute_diff(self._proposal(tmp_path, "same\n", "same\n"))
        # unified_diff of identical content has no +/- lines
        assert not any(line.startswith(("+", "-")) and not line.startswith("---") and not line.startswith("+++")
                       for line in diff.splitlines())


# ── apply_proposal ────────────────────────────────────────────────────────────


class TestApplyProposal:
    def _setup(self, tmp_path: Path, content: str):
        """Real file on disk + session with that file seeded in context."""
        session = Session(model="test", working_dir=tmp_path, max_tokens=100_000)
        path = _seed(session, tmp_path, "foo.py", content)
        return session, path

    # search-not-found

    def test_search_not_found_returns_false(self, tmp_path):
        session, path = self._setup(tmp_path, "actual\n")
        result = apply_proposal(
            EditProposal(path=path, search="missing\n", replace="new\n"),
            session, _mock_renderer(), _mock_reader(),
        )
        assert result is False

    def test_search_not_found_prints_error(self, tmp_path):
        session, path = self._setup(tmp_path, "actual\n")
        renderer = _mock_renderer()
        apply_proposal(
            EditProposal(path=path, search="missing\n", replace="new\n"),
            session, renderer, _mock_reader(),
        )
        renderer.print_error.assert_called()

    def test_search_not_found_does_not_write_disk(self, tmp_path):
        session, path = self._setup(tmp_path, "original\n")
        apply_proposal(
            EditProposal(path=path, search="missing\n", replace="new\n"),
            session, _mock_renderer(), _mock_reader(),
        )
        assert path.read_text() == "original\n"

    # ambiguity

    def test_ambiguous_search_returns_false(self, tmp_path):
        session, path = self._setup(tmp_path, "dup\ndup\n")
        result = apply_proposal(
            EditProposal(path=path, search="dup\n", replace="unique\n"),
            session, _mock_renderer(), _mock_reader(),
        )
        assert result is False

    def test_ambiguous_search_prints_error(self, tmp_path):
        session, path = self._setup(tmp_path, "dup\ndup\n")
        renderer = _mock_renderer()
        apply_proposal(
            EditProposal(path=path, search="dup\n", replace="unique\n"),
            session, renderer, _mock_reader(),
        )
        renderer.print_error.assert_called()

    def test_ambiguous_search_does_not_write_disk(self, tmp_path):
        content = "dup\ndup\n"
        session, path = self._setup(tmp_path, content)
        apply_proposal(
            EditProposal(path=path, search="dup\n", replace="unique\n"),
            session, _mock_renderer(), _mock_reader(),
        )
        assert path.read_text() == content

    # confirmation — accepted

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", "Yes"])
    def test_accepted_returns_true(self, tmp_path, answer):
        session, path = self._setup(tmp_path, "old\n")
        result = apply_proposal(
            EditProposal(path=path, search="old\n", replace="new\n"),
            session, _mock_renderer(), _mock_reader(answer),
        )
        assert result is True

    def test_accepted_writes_correct_content(self, tmp_path):
        session, path = self._setup(tmp_path, "old content\n")
        apply_proposal(
            EditProposal(path=path, search="old content\n", replace="new content\n"),
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert path.read_text() == "new content\n"

    def test_accepted_replaces_only_within_larger_file(self, tmp_path):
        content = "prefix\nTARGET\nsuffix\n"
        session, path = self._setup(tmp_path, content)
        apply_proposal(
            EditProposal(path=path, search="TARGET\n", replace="REPLACED\n"),
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert path.read_text() == "prefix\nREPLACED\nsuffix\n"

    # confirmation — declined

    @pytest.mark.parametrize("answer", ["n", "N", "no", "", "nope"])
    def test_declined_returns_false(self, tmp_path, answer):
        session, path = self._setup(tmp_path, "old\n")
        result = apply_proposal(
            EditProposal(path=path, search="old\n", replace="new\n"),
            session, _mock_renderer(), _mock_reader(answer),
        )
        assert result is False

    def test_declined_does_not_write_disk(self, tmp_path):
        session, path = self._setup(tmp_path, "original\n")
        apply_proposal(
            EditProposal(path=path, search="original\n", replace="changed\n"),
            session, _mock_renderer(), _mock_reader("n"),
        )
        assert path.read_text() == "original\n"

    # diff display

    def test_diff_displayed_before_confirmation(self, tmp_path):
        session, path = self._setup(tmp_path, "old\n")
        renderer = _mock_renderer()
        apply_proposal(
            EditProposal(path=path, search="old\n", replace="new\n"),
            session, renderer, _mock_reader("y"),
        )
        renderer.print_info.assert_called()

    # session isolation

    def test_context_files_not_updated_after_apply(self, tmp_path):
        session, path = self._setup(tmp_path, "old\n")
        apply_proposal(
            EditProposal(path=path, search="old\n", replace="new\n"),
            session, _mock_renderer(), _mock_reader("y"),
        )
        # refresh_files() should pick this up next turn — not apply_proposal's job
        assert session.context_files[path].content == "old\n"

    # prompt content

    def test_file_without_trailing_newline_is_matched(self, tmp_path):
        # Regression: parse_proposals always produces search/replace with a
        # structural trailing \n.  When the file's last line has no \n, a direct
        # count() would return 0.  apply_proposal must fall back to stripping \n
        # from both search and replace before matching — and must NOT add a
        # trailing \n to the written file content.
        content = "print('hello world')"          # no trailing newline
        session, path = self._setup(tmp_path, content)
        result = apply_proposal(
            EditProposal(
                path=path,
                search="print('hello world')\n",    # structural \n from parse_proposals
                replace="print('hello world!')\n",
            ),
            session, _mock_renderer(), _mock_reader("y"),
        )
        assert result is True
        assert path.read_text() == "print('hello world!')"  # no trailing \n added

    def test_prompt_mentions_filename(self, tmp_path):
        session, path = self._setup(tmp_path, "old\n")
        reader = _mock_reader("y")
        apply_proposal(
            EditProposal(path=path, search="old\n", replace="new\n"),
            session, _mock_renderer(), reader,
        )
        prompt = reader.read_line.call_args.args[0]
        assert "foo.py" in prompt


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
