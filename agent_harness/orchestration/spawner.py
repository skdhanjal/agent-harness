"""Adaptive Orchestration -- sub-agent spawning half.

Runs independent units of work in parallel, each as its own full
harness instance, capped by max_workers so a bug can't fork-bomb its
way into an unbounded API bill.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DagNode:
    id: str
    task: str


class SubAgentRunner(Protocol):
    def run(self, task: str) -> object: ...


class SubAgentSpawner:
    def __init__(self, harness_factory: Callable[[], SubAgentRunner], max_workers: int = 4) -> None:
        self._harness_factory = harness_factory
        self._max_workers = max_workers

    def run_dag(self, nodes: list[DagNode]) -> dict[str, object]:
        results: dict[str, object] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._run_node, node): node for node in nodes}
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                results[node.id] = future.result()
        return results

    def _run_node(self, node: DagNode) -> object:
        agent = self._harness_factory()
        return agent.run(node.task)
