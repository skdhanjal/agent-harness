"""Durable state: facts written to disk, read-through/write-through.

Always reads fresh from disk and writes immediately -- no in-memory
cache to go stale. Simple over fast, since correctness after a crash
is the entire point of this module.
"""

from __future__ import annotations

import json
from pathlib import Path


class DurableStateStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            self._path.write_text("{}")

    def _read(self) -> dict[str, object]:
        data: dict[str, object] = json.loads(self._path.read_text())
        return data

    def get(self, key: str) -> object | None:
        return self._read().get(key)

    def set(self, key: str, value: object) -> None:
        data = self._read()
        data[key] = value
        self._path.write_text(json.dumps(data))
