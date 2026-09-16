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
    difficulty: int = 0  # fed to ModelRouter.choose() by a driver; unused by the spawner itself


class SubAgentRunner(Protocol):
    def run(self, task: str) -> object: ...


@dataclass(frozen=True)
class NodeResult:
    """Mirrors ToolExecutionResult's ok/result/error shape (tools/registry.py)
    -- one node raising shouldn't take the rest of the DAG's results with it.
    """

    ok: bool
    result: object = None
    error: str | None = None


class SubAgentSpawner:
    def __init__(
        self, harness_factory: Callable[[DagNode], SubAgentRunner], max_workers: int = 4
    ) -> None:
        self._harness_factory = harness_factory
        self._max_workers = max_workers

    def run_dag(self, nodes: list[DagNode]) -> dict[str, NodeResult]:
        results: dict[str, NodeResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._run_node, node): node for node in nodes}
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                results[node.id] = future.result()
        return results

    def _run_node(self, node: DagNode) -> NodeResult:
        agent = self._harness_factory(node)
        try:
            return NodeResult(ok=True, result=agent.run(node.task))
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
