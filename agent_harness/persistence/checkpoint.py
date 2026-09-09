"""Checkpointing: enough state saved to resume a run instead of restarting."""

from __future__ import annotations

import json
from pathlib import Path


def save_checkpoint(
    run_id: str, state: dict[str, object], directory: str | Path = "./checkpoints"
) -> None:
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{run_id}.json").write_text(json.dumps(state))


def load_checkpoint(
    run_id: str, directory: str | Path = "./checkpoints"
) -> dict[str, object] | None:
    path = Path(directory) / f"{run_id}.json"
    if not path.exists():
        return None
    data: dict[str, object] = json.loads(path.read_text())
    return data
