"""Tracing: a ledger of every event, so debugging and cost tracking
don't depend on memory of what happened."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TraceEvent:
    run_id: str
    # "llm_call" | "tool_call" | "transition" | "gate_decision" | "context_compaction"
    event_type: str
    payload: dict[str, object]
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ts: float = field(default_factory=time.time)


class TraceLedger:
    def __init__(self, path: str | Path = "./traces.jsonl") -> None:
        self._path = Path(path)

    def log(self, event: TraceEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def total_cost(self, run_id: str) -> float:
        if not self._path.exists():
            return 0.0
        total = 0.0
        for line in self._path.read_text().splitlines():
            record = json.loads(line)
            if record["run_id"] == run_id:
                total += record["cost_usd"]
        return total
