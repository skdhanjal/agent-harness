"""Verification tests for the expanded filesystem tool suite.

Proves: every tool's happy path works relative to a sandbox root, and the
sandbox actually blocks path-traversal, absolute-outside-root, and
symlink escapes -- rather than silently touching disk outside the root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    reg = ToolRegistry()
    register_filesystem_tools(reg, root=tmp_path)
    return reg


def test_write_then_read_round_trips(registry: ToolRegistry) -> None:
    write_result = registry.execute("write_file", {"path": "a.md", "content": "hello"})
    assert write_result.ok is True

    read_result = registry.execute("read_file", {"path": "a.md"})
    assert read_result.ok is True
    assert read_result.result == "hello"


def test_read_file_missing_returns_structured_error(registry: ToolRegistry) -> None:
    result = registry.execute("read_file", {"path": "nope.txt"})

    assert result.ok is False
    assert result.error is not None
    assert "execution_error" in result.error


def test_read_file_truncates_large_content(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "big.txt", "content": "x" * 100})

    result = registry.execute("read_file", {"path": "big.txt", "max_chars": 10})

    assert result.ok is True
    assert isinstance(result.result, str)
    assert result.result.startswith("x" * 10)
    assert "truncated" in result.result


def test_list_files_empty_for_missing_directory(registry: ToolRegistry) -> None:
    result = registry.execute("list_files", {"directory": "does_not_exist"})

    assert result.ok is True
    assert result.result == []


def test_list_files_recursive_finds_nested_entries(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "top.txt", "content": "t"})
    registry.execute("write_file", {"path": "nested/deep.txt", "content": "d"})

    top_level = registry.execute("list_files", {})
    assert top_level.ok is True
    assert "top.txt" in top_level.result  # type: ignore[operator]
    assert "nested/deep.txt" not in top_level.result  # type: ignore[operator]

    recursive = registry.execute("list_files", {"recursive": True})
    assert recursive.ok is True
    assert "nested/deep.txt" in recursive.result  # type: ignore[operator]


def test_glob_files_matches_pattern(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "src/a.py", "content": "1"})
    registry.execute("write_file", {"path": "src/b.py", "content": "2"})
    registry.execute("write_file", {"path": "src/c.md", "content": "3"})

    result = registry.execute("glob_files", {"pattern": "**/*.py"})

    assert result.ok is True
    assert sorted(result.result) == ["src/a.py", "src/b.py"]  # type: ignore[type-var]


def test_glob_files_pattern_cannot_escape_root(registry: ToolRegistry, tmp_path: Path) -> None:
    sibling = tmp_path.parent / "secret_sibling.txt"
    sibling.write_text("shh")
    try:
        result = registry.execute("glob_files", {"pattern": "../*.txt"})
        assert result.ok is True
        assert result.result == []
    finally:
        sibling.unlink()


def test_grep_files_finds_matching_lines(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "a.txt", "content": "hello world\nbye"})
    registry.execute("write_file", {"path": "b.txt", "content": "another hello"})

    result = registry.execute("grep_files", {"pattern": "hello"})

    assert result.ok is True
    assert len(result.result) == 2  # type: ignore[arg-type]
    paths = {m["path"] for m in result.result}  # type: ignore[union-attr]
    assert paths == {"a.txt", "b.txt"}


def test_grep_files_regex_and_case_insensitive(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "a.txt", "content": "Error: boom\nfine"})

    result = registry.execute(
        "grep_files", {"pattern": "error", "regex": True, "case_sensitive": False}
    )

    assert result.ok is True
    assert len(result.result) == 1  # type: ignore[arg-type]
    assert result.result[0]["text"] == "Error: boom"  # type: ignore[index]


def test_grep_files_rejects_max_matches_below_one(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "a.txt", "content": "hello"})

    result = registry.execute("grep_files", {"pattern": "hello", "max_matches": 0})

    assert result.ok is False
    assert result.error is not None
    assert "max_matches" in result.error


def test_grep_files_accepts_a_single_file_as_directory(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "a.txt", "content": "hello world"})
    registry.execute("write_file", {"path": "b.txt", "content": "hello again"})

    result = registry.execute("grep_files", {"pattern": "hello", "directory": "a.txt"})

    assert result.ok is True
    assert len(result.result) == 1  # type: ignore[arg-type]
    assert result.result[0]["path"] == "a.txt"  # type: ignore[index]


def test_grep_files_missing_path_returns_empty_list(registry: ToolRegistry) -> None:
    result = registry.execute("grep_files", {"pattern": "hello", "directory": "does_not_exist"})

    assert result.ok is True
    assert result.result == []


def test_delete_file_removes_file(registry: ToolRegistry, tmp_path: Path) -> None:
    registry.execute("write_file", {"path": "gone.txt", "content": "x"})

    result = registry.execute("delete_file", {"path": "gone.txt"})

    assert result.ok is True
    assert not (tmp_path / "gone.txt").exists()


def test_delete_file_missing_returns_structured_error(registry: ToolRegistry) -> None:
    result = registry.execute("delete_file", {"path": "nope.txt"})

    assert result.ok is False
    assert result.error is not None


def test_move_file_relocates_file(registry: ToolRegistry, tmp_path: Path) -> None:
    registry.execute("write_file", {"path": "src.txt", "content": "content"})

    result = registry.execute("move_file", {"src": "src.txt", "dst": "dst/renamed.txt"})

    assert result.ok is True
    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dst" / "renamed.txt").read_text() == "content"


def test_move_file_refuses_overwrite_by_default(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "src.txt", "content": "new"})
    registry.execute("write_file", {"path": "dst.txt", "content": "old"})

    result = registry.execute("move_file", {"src": "src.txt", "dst": "dst.txt"})

    assert result.ok is False


def test_move_file_overwrite_flag_replaces_destination(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    registry.execute("write_file", {"path": "src.txt", "content": "new"})
    registry.execute("write_file", {"path": "dst.txt", "content": "old"})

    result = registry.execute("move_file", {"src": "src.txt", "dst": "dst.txt", "overwrite": True})

    assert result.ok is True
    assert (tmp_path / "dst.txt").read_text() == "new"


def test_make_directory_creates_nested_dirs(registry: ToolRegistry, tmp_path: Path) -> None:
    result = registry.execute("make_directory", {"path": "a/b/c"})

    assert result.ok is True
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_file_stat_reports_metadata(registry: ToolRegistry) -> None:
    registry.execute("write_file", {"path": "a.txt", "content": "line1\nline2\nline3"})

    result = registry.execute("file_stat", {"path": "a.txt"})

    assert result.ok is True
    assert result.result["exists"] is True  # type: ignore[index]
    assert result.result["is_file"] is True  # type: ignore[index]
    assert result.result["size_bytes"] == 17  # type: ignore[index]
    assert result.result["line_count"] == 3  # type: ignore[index]


def test_file_stat_missing_path_is_not_an_error(registry: ToolRegistry) -> None:
    result = registry.execute("file_stat", {"path": "nope.txt"})

    assert result.ok is True
    assert result.result == {"exists": False, "path": "nope.txt"}


def test_file_stat_line_count_none_for_directory(registry: ToolRegistry) -> None:
    registry.execute("make_directory", {"path": "a_dir"})

    result = registry.execute("file_stat", {"path": "a_dir"})

    assert result.ok is True
    assert result.result["is_dir"] is True  # type: ignore[index]
    assert result.result["line_count"] is None  # type: ignore[index]


def test_file_stat_line_count_none_for_undecodable_file(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe\x00\x01\x02")

    result = registry.execute("file_stat", {"path": "binary.dat"})

    assert result.ok is True
    assert result.result["exists"] is True  # type: ignore[index]
    assert result.result["line_count"] is None  # type: ignore[index]


def test_sandbox_blocks_relative_traversal_escape(registry: ToolRegistry, tmp_path: Path) -> None:
    secret = tmp_path.parent / "escape_target.txt"
    secret.write_text("secret")
    try:
        result = registry.execute("read_file", {"path": f"../{secret.name}"})
        assert result.ok is False
        assert result.error is not None
    finally:
        secret.unlink()


def test_sandbox_blocks_absolute_path_outside_root(registry: ToolRegistry, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope")
    try:
        result = registry.execute("read_file", {"path": str(outside)})
        assert result.ok is False
    finally:
        outside.unlink()


def test_sandbox_allows_absolute_path_inside_root(registry: ToolRegistry, tmp_path: Path) -> None:
    inside = tmp_path / "inside.txt"

    result = registry.execute("write_file", {"path": str(inside), "content": "ok"})

    assert result.ok is True
    assert inside.read_text() == "ok"


def test_sandbox_blocks_symlink_escape(registry: ToolRegistry, tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret")
    link = tmp_path / "escape_link"
    try:
        os.symlink(outside_dir, link)
        result = registry.execute("read_file", {"path": "escape_link/secret.txt"})
        assert result.ok is False
    finally:
        link.unlink()
        (outside_dir / "secret.txt").unlink()
        outside_dir.rmdir()


def test_all_nine_tools_are_registered(registry: ToolRegistry) -> None:
    names = {s["function"]["name"] for s in registry.schema_for_llm()}  # type: ignore[index]

    assert names == {
        "read_file",
        "write_file",
        "list_files",
        "glob_files",
        "grep_files",
        "file_stat",
        "make_directory",
        "move_file",
        "delete_file",
    }
