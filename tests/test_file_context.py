import os
from pathlib import Path

import pytest

from fizzy.session import FileEntry, Session
from fizzy import file_context
from fizzy.file_context import MAX_FILE_BYTES


# ── helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def session(tmp_path):
    return Session(model="test-model", working_dir=tmp_path, max_tokens=100_000)


def make_text_file(directory: Path, name: str, content: str = "hello\n") -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ── add_file ──────────────────────────────────────────────────────────────────


class TestAddFile:
    def test_adds_valid_text_file(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py", "x = 1\n")
        ok, _ = file_context.add_file(session, "foo.py")
        assert ok is True
        assert (tmp_path / "foo.py").resolve() in session.context_files

    def test_returns_success_message_containing_filename(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        ok, msg = file_context.add_file(session, "foo.py")
        assert ok is True
        assert "foo.py" in msg

    def test_stores_correct_content(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py", "x = 42\n")
        file_context.add_file(session, "foo.py")
        resolved = (tmp_path / "foo.py").resolve()
        assert session.context_files[resolved].content == "x = 42\n"

    def test_stores_mtime(self, session, tmp_path):
        p = make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        resolved = p.resolve()
        assert session.context_files[resolved].mtime == pytest.approx(p.stat().st_mtime)

    def test_resolves_path_relative_to_working_dir(self, session, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        make_text_file(subdir, "bar.py", "y = 2\n")
        ok, _ = file_context.add_file(session, "src/bar.py")
        assert ok is True
        assert (tmp_path / "src" / "bar.py").resolve() in session.context_files

    def test_already_tracked_is_noop(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        ok, msg = file_context.add_file(session, "foo.py")
        assert ok is False
        assert len(session.context_files) == 1

    def test_already_tracked_message_mentions_already(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        _, msg = file_context.add_file(session, "foo.py")
        assert "already" in msg.lower()

    def test_file_not_found_returns_false(self, session):
        ok, msg = file_context.add_file(session, "nonexistent.py")
        assert ok is False
        assert "nonexistent.py" in msg

    def test_binary_file_returns_false(self, session, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(bytes(range(256)))
        ok, msg = file_context.add_file(session, "image.png")
        assert ok is False
        assert "binary" in msg.lower()

    def test_binary_file_not_added_to_context(self, session, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(bytes(range(256)))
        file_context.add_file(session, "image.png")
        assert p.resolve() not in session.context_files

    def test_non_utf8_file_returns_false(self, session, tmp_path):
        p = tmp_path / "latin.txt"
        p.write_bytes(b"\xff\xfe\xe9\xe0\xf6")  # not valid UTF-8
        ok, msg = file_context.add_file(session, "latin.txt")
        assert ok is False

    def test_non_utf8_file_not_added_to_context(self, session, tmp_path):
        p = tmp_path / "latin.txt"
        p.write_bytes(b"\xff\xfe\xe9\xe0\xf6")
        file_context.add_file(session, "latin.txt")
        assert p.resolve() not in session.context_files

    def test_oversized_file_returns_false(self, session, tmp_path):
        p = tmp_path / "big.txt"
        p.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        ok, msg = file_context.add_file(session, "big.txt")
        assert ok is False
        assert "big.txt" in msg

    def test_oversized_file_message_mentions_limit(self, session, tmp_path):
        p = tmp_path / "big.txt"
        p.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        _, msg = file_context.add_file(session, "big.txt")
        assert "512" in msg  # size limit in KB

    def test_oversized_file_not_added_to_context(self, session, tmp_path):
        p = tmp_path / "big.txt"
        p.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        file_context.add_file(session, "big.txt")
        assert p.resolve() not in session.context_files

    def test_file_at_exact_size_limit_is_allowed(self, session, tmp_path):
        p = tmp_path / "exact.txt"
        p.write_bytes(b"a" * MAX_FILE_BYTES)
        ok, _ = file_context.add_file(session, "exact.txt")
        assert ok is True

    def test_empty_path_string_returns_false(self, session):
        ok, _ = file_context.add_file(session, "")
        assert ok is False

    def test_directory_path_returns_false(self, session, tmp_path):
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        ok, _ = file_context.add_file(session, "mydir")
        assert ok is False


# ── drop_file ─────────────────────────────────────────────────────────────────


class TestDropFile:
    def test_removes_tracked_file(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        ok, _ = file_context.drop_file(session, "foo.py")
        assert ok is True
        assert (tmp_path / "foo.py").resolve() not in session.context_files

    def test_returns_success_message_containing_filename(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        ok, msg = file_context.drop_file(session, "foo.py")
        assert ok is True
        assert "foo.py" in msg

    def test_not_tracked_returns_false(self, session):
        ok, msg = file_context.drop_file(session, "ghost.py")
        assert ok is False
        assert "ghost.py" in msg

    def test_resolves_path_relative_to_working_dir(self, session, tmp_path):
        subdir = tmp_path / "lib"
        subdir.mkdir()
        make_text_file(subdir, "util.py")
        file_context.add_file(session, "lib/util.py")
        ok, _ = file_context.drop_file(session, "lib/util.py")
        assert ok is True
        assert (tmp_path / "lib" / "util.py").resolve() not in session.context_files

    def test_drop_leaves_other_files_intact(self, session, tmp_path):
        make_text_file(tmp_path, "a.py")
        make_text_file(tmp_path, "b.py")
        file_context.add_file(session, "a.py")
        file_context.add_file(session, "b.py")
        file_context.drop_file(session, "a.py")
        assert (tmp_path / "b.py").resolve() in session.context_files


# ── list_files ────────────────────────────────────────────────────────────────


class TestListFiles:
    def test_empty_when_no_files_tracked(self, session):
        assert file_context.list_files(session) == []

    def test_returns_tracked_paths(self, session, tmp_path):
        make_text_file(tmp_path, "a.py")
        make_text_file(tmp_path, "b.py")
        file_context.add_file(session, "a.py")
        file_context.add_file(session, "b.py")
        result = file_context.list_files(session)
        assert (tmp_path / "a.py").resolve() in result
        assert (tmp_path / "b.py").resolve() in result

    def test_returns_list_of_path_objects(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        result = file_context.list_files(session)
        assert all(isinstance(p, Path) for p in result)

    def test_insertion_order_preserved(self, session, tmp_path):
        for name in ("first.py", "second.py", "third.py"):
            make_text_file(tmp_path, name)
            file_context.add_file(session, name)
        names = [p.name for p in file_context.list_files(session)]
        assert names == ["first.py", "second.py", "third.py"]

    def test_dropped_file_absent_from_list(self, session, tmp_path):
        make_text_file(tmp_path, "a.py")
        make_text_file(tmp_path, "b.py")
        file_context.add_file(session, "a.py")
        file_context.add_file(session, "b.py")
        file_context.drop_file(session, "a.py")
        names = [p.name for p in file_context.list_files(session)]
        assert "a.py" not in names
        assert "b.py" in names


# ── refresh_files ─────────────────────────────────────────────────────────────


class TestRefreshFiles:
    def test_no_files_returns_empty_list(self, session):
        assert file_context.refresh_files(session) == []

    def test_unchanged_file_produces_no_notices(self, session, tmp_path):
        make_text_file(tmp_path, "stable.py", "original\n")
        file_context.add_file(session, "stable.py")
        result = file_context.refresh_files(session)
        assert result == []

    def test_unchanged_file_content_not_altered(self, session, tmp_path):
        make_text_file(tmp_path, "stable.py", "original\n")
        file_context.add_file(session, "stable.py")
        resolved = (tmp_path / "stable.py").resolve()
        file_context.refresh_files(session)
        assert session.context_files[resolved].content == "original\n"

    def test_changed_file_produces_one_notice(self, session, tmp_path):
        p = make_text_file(tmp_path, "changing.py", "v1\n")
        file_context.add_file(session, "changing.py")
        p.write_text("v2\n", encoding="utf-8")
        future_mtime = p.stat().st_mtime + 10.0
        os.utime(p, (p.stat().st_atime, future_mtime))
        result = file_context.refresh_files(session)
        assert len(result) == 1

    def test_changed_file_notice_contains_filename(self, session, tmp_path):
        p = make_text_file(tmp_path, "changing.py", "v1\n")
        file_context.add_file(session, "changing.py")
        p.write_text("v2\n", encoding="utf-8")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10.0))
        result = file_context.refresh_files(session)
        assert "changing.py" in result[0]

    def test_changed_file_content_is_updated(self, session, tmp_path):
        p = make_text_file(tmp_path, "changing.py", "v1\n")
        file_context.add_file(session, "changing.py")
        resolved = p.resolve()
        p.write_text("v2\n", encoding="utf-8")
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10.0))
        file_context.refresh_files(session)
        assert session.context_files[resolved].content == "v2\n"

    def test_changed_file_mtime_is_updated(self, session, tmp_path):
        p = make_text_file(tmp_path, "foo.py", "old\n")
        file_context.add_file(session, "foo.py")
        resolved = p.resolve()
        new_mtime = p.stat().st_mtime + 10.0
        os.utime(p, (p.stat().st_atime, new_mtime))
        p.write_text("new\n", encoding="utf-8")
        os.utime(p, (p.stat().st_atime, new_mtime))
        file_context.refresh_files(session)
        assert session.context_files[resolved].mtime == pytest.approx(new_mtime)

    def test_deleted_file_is_removed_from_context(self, session, tmp_path):
        p = make_text_file(tmp_path, "gone.py")
        file_context.add_file(session, "gone.py")
        resolved = p.resolve()
        p.unlink()
        file_context.refresh_files(session)
        assert resolved not in session.context_files

    def test_deleted_file_produces_notice(self, session, tmp_path):
        p = make_text_file(tmp_path, "gone.py")
        file_context.add_file(session, "gone.py")
        p.unlink()
        result = file_context.refresh_files(session)
        assert len(result) == 1
        assert "gone.py" in result[0]

    def test_only_changed_files_appear_in_notices(self, session, tmp_path):
        make_text_file(tmp_path, "stable.py", "no change\n")
        p_changed = make_text_file(tmp_path, "changed.py", "before\n")
        file_context.add_file(session, "stable.py")
        file_context.add_file(session, "changed.py")
        new_mtime = p_changed.stat().st_mtime + 10.0
        p_changed.write_text("after\n", encoding="utf-8")
        os.utime(p_changed, (p_changed.stat().st_atime, new_mtime))
        result = file_context.refresh_files(session)
        assert len(result) == 1
        assert "changed.py" in result[0]

    def test_multiple_changed_files_all_reported(self, session, tmp_path):
        files = [make_text_file(tmp_path, f"f{i}.py", f"v{i}\n") for i in range(3)]
        for i in range(3):
            file_context.add_file(session, f"f{i}.py")
        for p in files:
            p.write_text("updated\n", encoding="utf-8")
            os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10.0))
        result = file_context.refresh_files(session)
        assert len(result) == 3


# ── build_system_message ──────────────────────────────────────────────────────


class TestBuildSystemMessage:
    def test_no_files_returns_none(self, session):
        assert file_context.build_system_message(session) is None

    def test_returns_dict_with_system_role(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py", "x = 1\n")
        file_context.add_file(session, "foo.py")
        msg = file_context.build_system_message(session)
        assert isinstance(msg, dict)
        assert msg["role"] == "system"

    def test_returns_dict_with_content_key(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        msg = file_context.build_system_message(session)
        assert "content" in msg

    def test_content_is_a_string(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        msg = file_context.build_system_message(session)
        assert isinstance(msg["content"], str)

    def test_content_includes_filename(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py", "x = 1\n")
        file_context.add_file(session, "foo.py")
        msg = file_context.build_system_message(session)
        assert "foo.py" in msg["content"]

    def test_content_includes_file_body(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py", "UNIQUE_MARKER = True\n")
        file_context.add_file(session, "foo.py")
        msg = file_context.build_system_message(session)
        assert "UNIQUE_MARKER = True" in msg["content"]

    def test_multiple_files_all_present_in_content(self, session, tmp_path):
        make_text_file(tmp_path, "alpha.py", "alpha body\n")
        make_text_file(tmp_path, "beta.py", "beta body\n")
        file_context.add_file(session, "alpha.py")
        file_context.add_file(session, "beta.py")
        msg = file_context.build_system_message(session)
        assert "alpha.py" in msg["content"]
        assert "alpha body" in msg["content"]
        assert "beta.py" in msg["content"]
        assert "beta body" in msg["content"]

    def test_does_not_mutate_session_history(self, session, tmp_path):
        make_text_file(tmp_path, "foo.py")
        file_context.add_file(session, "foo.py")
        file_context.build_system_message(session)
        assert session.history == []
