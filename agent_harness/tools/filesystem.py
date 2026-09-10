"""Filesystem tools: reads/list/stat/mkdir are safe, writes need a gate.

Risk tiers aren't decided here -- that's up to whoever wires up the
harness. Every path is resolved through a PathSandbox first.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.tools.registry import ToolRegistry
from agent_harness.tools.sandbox import PathSandbox


def register_filesystem_tools(registry: ToolRegistry, root: str | Path = ".") -> None:
    sandbox = PathSandbox(root)

    @registry.tool
    def read_file(path: str, max_chars: int = 20_000) -> str:
        """Read a text file's contents. `path` is relative to the sandbox
        root. Truncates to `max_chars` characters (default 20000); a
        '...[truncated]' marker means there was more.
        """
        target = sandbox.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        text = target.read_text()
        if len(text) > max_chars:
            return text[:max_chars] + f"\n...[truncated: showing {max_chars} of {len(text)} chars]"
        return text

    @registry.tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed
        and overwriting any existing file. `path` is relative to the
        sandbox root.
        """
        target = sandbox.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @registry.tool
    def list_files(directory: str = ".", recursive: bool = False) -> list[str]:
        """List entries in `directory` (default: sandbox root), or an empty
        list if it doesn't exist. Paths are relative to the sandbox root,
        sorted. `recursive=True` descends into subdirectories.
        """
        target = sandbox.resolve(directory)
        if not target.is_dir():
            return []
        entries = target.rglob("*") if recursive else target.iterdir()
        return sorted(
            str(p.relative_to(sandbox.root)) for p in entries if sandbox.contains(p.resolve())
        )

    @registry.tool
    def glob_files(pattern: str) -> list[str]:
        """Find files matching a glob pattern under the sandbox root, e.g.
        '**/*.py'. Returns file paths (not directories), relative to the
        sandbox root, sorted.
        """
        matches: list[str] = []
        for p in sandbox.root.glob(pattern):
            if not p.is_file():
                continue
            if not sandbox.contains(p.resolve()):
                continue  # escaped the sandbox via pattern traversal
            matches.append(str(p.relative_to(sandbox.root)))
        return sorted(matches)

    @registry.tool
    def grep_files(
        pattern: str,
        directory: str = ".",
        regex: bool = False,
        case_sensitive: bool = True,
        max_matches: int = 200,
    ) -> list[dict[str, object]]:
        """Search file contents under `directory` (recursively, default:
        sandbox root) for `pattern`. `directory` may also be a single file,
        to search just that file. Substring match by default; `regex=True`
        treats it as a regular expression. Skips files that can't be decoded
        as text. `max_matches` must be >= 1. Returns up to `max_matches`
        {"path", "line", "text"} dicts.
        """
        if max_matches < 1:
            raise ValueError(f"max_matches must be >= 1, got {max_matches}")

        base = sandbox.resolve(directory)
        if base.is_file():
            candidates: list[Path] = [base]
        elif base.is_dir():
            candidates = sorted(base.rglob("*"))
        else:
            return []
        matcher = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE) if regex else None
        needle = pattern if case_sensitive else pattern.lower()

        matches: list[dict[str, object]] = []
        for file_path in candidates:
            if len(matches) >= max_matches:
                break
            if not file_path.is_file() or not sandbox.contains(file_path.resolve()):
                continue
            try:
                lines = file_path.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(lines, start=1):
                if len(matches) >= max_matches:
                    break
                haystack = line if case_sensitive else line.lower()
                hit = matcher.search(line) if matcher else needle in haystack
                if hit:
                    matches.append(
                        {
                            "path": str(file_path.relative_to(sandbox.root)),
                            "line": lineno,
                            "text": line,
                        }
                    )
        return matches

    @registry.tool
    def file_stat(path: str) -> dict[str, object]:
        """Metadata for `path`. Returns {"exists": False, "path": path} if
        nothing is there. Otherwise includes 'is_file', 'is_dir',
        'size_bytes', 'modified_at' (UTC), and for files, 'line_count'
        (None if the file can't be decoded as text) -- use this instead of
        counting lines yourself from read_file's output.
        """
        target = sandbox.resolve(path)
        if not target.exists():
            return {"exists": False, "path": path}
        st = target.stat()
        line_count: int | None = None
        if target.is_file():
            try:
                line_count = len(target.read_text().splitlines())
            except (UnicodeDecodeError, OSError):
                line_count = None
        return {
            "exists": True,
            "path": path,
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "size_bytes": st.st_size,
            "modified_at": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            "line_count": line_count,
        }

    @registry.tool
    def make_directory(path: str) -> str:
        """Create a directory (and parents) at `path`. No-op if it exists."""
        target = sandbox.resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"created directory {path}"

    @registry.tool
    def move_file(src: str, dst: str, overwrite: bool = False) -> str:
        """Move/rename `src` to `dst`. Fails if `dst` already exists unless
        `overwrite=True`.
        """
        src_path = sandbox.resolve(src)
        dst_path = sandbox.resolve(dst)
        if not src_path.is_file():
            raise FileNotFoundError(f"no such file: {src}")
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"'{dst}' already exists (pass overwrite=True to replace it)")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.replace(dst_path)
        return f"moved {src} -> {dst}"

    @registry.tool
    def delete_file(path: str) -> str:
        """Delete the file at `path`. Fails if it doesn't exist or is a
        directory.
        """
        target = sandbox.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        target.unlink()
        return f"deleted {path}"
