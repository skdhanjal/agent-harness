"""Path sandboxing for filesystem tools."""

from __future__ import annotations

from pathlib import Path


class SandboxViolationError(ValueError):
    """Raised when a path would resolve outside the sandbox root."""


class PathSandbox:
    """Resolves paths against a fixed root, rejecting anything that escapes it."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, raw_path: str) -> Path:
        """Resolve `raw_path` against root; raise SandboxViolationError on escape."""
        candidate = Path(raw_path)
        joined = candidate if candidate.is_absolute() else self.root / candidate
        resolved = joined.resolve()  # normalizes ".." and follows symlinks
        if not self.contains(resolved):
            raise SandboxViolationError(
                f"path '{raw_path}' resolves to '{resolved}', which is outside "
                f"sandbox root '{self.root}'"
            )
        return resolved

    def contains(self, resolved: Path) -> bool:
        return resolved.is_relative_to(self.root)
