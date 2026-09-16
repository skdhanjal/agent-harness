"""Adaptive Orchestration -- the piece that actually composes the other two:
picks each node's starting model tier from its difficulty (ModelRouter),
runs the DAG in parallel (SubAgentSpawner), and retries a node that failed
one tier up instead of retrying blind at the same tier or giving up on the
first failure.

Task decomposition -- turning one task string into a DagNode list -- isn't
done here; the caller supplies the DAG. See
docs/production_readiness_todo.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_harness.orchestration.router import ModelRouter, ModelTier
from agent_harness.orchestration.spawner import DagNode, SubAgentRunner, SubAgentSpawner

HarnessBuilder = Callable[[DagNode, ModelTier], SubAgentRunner]


@dataclass(frozen=True)
class OrchestrationResult:
    ok: bool
    result: object = None
    error: str | None = None
    tier_used: str = ""
    attempts: int = 0


class OrchestrationDriver:
    def __init__(
        self,
        router: ModelRouter,
        harness_builder: HarnessBuilder,
        max_workers: int = 4,
        max_attempts: int = 3,
    ) -> None:
        self._router = router
        self._harness_builder = harness_builder
        self._max_workers = max_workers
        self._max_attempts = max_attempts

    def run(self, nodes: list[DagNode]) -> dict[str, OrchestrationResult]:
        tiers = {node.id: self._router.choose(node.difficulty) for node in nodes}
        attempts = {node.id: 1 for node in nodes}
        pending = list(nodes)
        final: dict[str, OrchestrationResult] = {}

        while pending:
            spawner = SubAgentSpawner(
                harness_factory=lambda node: self._harness_builder(node, tiers[node.id]),
                max_workers=self._max_workers,
            )
            batch = spawner.run_dag(pending)

            retry: list[DagNode] = []
            for node in pending:
                node_result = batch[node.id]
                if node_result.ok or attempts[node.id] >= self._max_attempts:
                    final[node.id] = OrchestrationResult(
                        ok=node_result.ok,
                        result=node_result.result,
                        error=node_result.error,
                        tier_used=tiers[node.id].name,
                        attempts=attempts[node.id],
                    )
                    continue

                tiers[node.id] = self._router.escalate(tiers[node.id])
                attempts[node.id] += 1
                retry.append(node)

            pending = retry

        return final
