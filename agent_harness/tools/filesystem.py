"""Reusable filesystem tools: reading/listing are safe, writing needs a gate.

These are ordinary Python functions decorated onto whatever ToolRegistry
the caller provides -- they don't know or care about risk tiers; that's
decided separately by whoever wires up the harness.
"""

from __future__ import annotations

from pathlib import Path

from agent_harness.tools.registry import ToolRegistry


def register_filesystem_tools(registry: ToolRegistry) -> None:
    @registry.tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    @registry.tool
    def list_files(directory: str) -> list[str]:
        """List files in a directory, or an empty list if it doesn't exist."""
        target = Path(directory)
        if not target.exists():
            return []
        return sorted(str(f) for f in target.iterdir())
