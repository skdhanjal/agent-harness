"""Durable record of approval decisions -- keyed by (run_id, action id), one
JSON file per run. Lets a resumed run tell "already decided, don't ask
again" from "never asked," and leaves evidence on disk of what a run was
waiting on even if it dies mid-approval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

Status = Literal["pending", "approved", "denied"]


def _path(run_id: str, directory: str | Path) -> Path:
    return Path(directory) / f"{run_id}.json"


def _load_all(run_id: str, directory: str | Path) -> dict[str, dict[str, object]]:
    path = _path(run_id, directory)
    if not path.exists():
        return {}
    data: dict[str, dict[str, object]] = json.loads(path.read_text())
    return data


def _write_all(run_id: str, directory: str | Path, records: dict[str, dict[str, object]]) -> None:
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    _path(run_id, directory).write_text(json.dumps(records))


def load_status(
    run_id: str, action_id: str, directory: str | Path = "./pending_actions"
) -> Status | None:
    record = _load_all(run_id, directory).get(action_id)
    if record is None:
        return None
    return cast(Status, record["status"])


def mark_pending(
    run_id: str,
    action_id: str,
    tool_name: str,
    args: dict[str, object],
    directory: str | Path = "./pending_actions",
) -> None:
    records = _load_all(run_id, directory)
    records[action_id] = {"status": "pending", "tool_name": tool_name, "args": args}
    _write_all(run_id, directory, records)


def resolve(
    run_id: str,
    action_id: str,
    approved: bool,
    directory: str | Path = "./pending_actions",
) -> None:
    records = _load_all(run_id, directory)
    records.setdefault(action_id, {})["status"] = "approved" if approved else "denied"
    _write_all(run_id, directory, records)
