"""Verification tests for OrchestrationDriver: the piece that actually
composes ModelRouter and SubAgentSpawner into a runnable multi-agent driver.

Proves: a node's starting tier comes from its difficulty, a failing node is
retried one tier up rather than at the same tier or not at all, a node that
keeps failing gives up after max_attempts instead of retrying forever, and
one node's failure doesn't stop a sibling node from completing.
"""

from __future__ import annotations

from agent_harness.orchestration.driver import OrchestrationDriver
from agent_harness.orchestration.router import ModelRouter, ModelTier
from agent_harness.orchestration.spawner import DagNode

_FAST = ModelTier("fast", cost_per_1k=0.001, capability_rank=0)
_FRONTIER = ModelTier("frontier", cost_per_1k=0.1, capability_rank=1)


class _ScriptedHarness:
    """Fails on every tier in `fails_on`, then succeeds."""

    def __init__(self, tier_name: str, fails_on: set[str]) -> None:
        self._tier_name = tier_name
        self._fails_on = fails_on

    def run(self, task: str) -> str:
        if self._tier_name in self._fails_on:
            raise RuntimeError(f"{self._tier_name} failed on {task}")
        return f"done:{task}@{self._tier_name}"


def _router() -> ModelRouter:
    return ModelRouter([_FAST, _FRONTIER])


def test_node_starts_at_the_tier_its_difficulty_maps_to() -> None:
    seen_tiers: list[str] = []

    def builder(node: DagNode, tier: ModelTier) -> _ScriptedHarness:
        seen_tiers.append(tier.name)
        return _ScriptedHarness(tier.name, fails_on=set())

    driver = OrchestrationDriver(router=_router(), harness_builder=builder)
    nodes = [
        DagNode(id="easy", task="t1", difficulty=0),
        DagNode(id="hard", task="t2", difficulty=1),
    ]

    results = driver.run(nodes)

    assert results["easy"].tier_used == "fast"
    assert results["hard"].tier_used == "frontier"
    assert results["easy"].ok is True
    assert results["easy"].attempts == 1


def test_failing_node_is_retried_one_tier_up_and_then_succeeds() -> None:
    def builder(node: DagNode, tier: ModelTier) -> _ScriptedHarness:
        return _ScriptedHarness(tier.name, fails_on={"fast"})

    driver = OrchestrationDriver(router=_router(), harness_builder=builder, max_attempts=3)
    nodes = [DagNode(id="n1", task="t1", difficulty=0)]

    results = driver.run(nodes)

    assert results["n1"].ok is True
    assert results["n1"].tier_used == "frontier"
    assert results["n1"].attempts == 2
    assert results["n1"].result == "done:t1@frontier"


def test_node_that_keeps_failing_gives_up_after_max_attempts() -> None:
    def builder(node: DagNode, tier: ModelTier) -> _ScriptedHarness:
        return _ScriptedHarness(tier.name, fails_on={"fast", "frontier"})

    driver = OrchestrationDriver(router=_router(), harness_builder=builder, max_attempts=2)
    nodes = [DagNode(id="n1", task="t1", difficulty=0)]

    results = driver.run(nodes)

    assert results["n1"].ok is False
    assert results["n1"].attempts == 2
    assert results["n1"].error is not None


def test_one_failing_node_does_not_block_a_sibling_from_completing() -> None:
    def builder(node: DagNode, tier: ModelTier) -> _ScriptedHarness:
        fails_on = {"fast", "frontier"} if node.id == "bad" else set()
        return _ScriptedHarness(tier.name, fails_on=fails_on)

    driver = OrchestrationDriver(router=_router(), harness_builder=builder, max_attempts=2)
    nodes = [DagNode(id="good", task="t1"), DagNode(id="bad", task="t2")]

    results = driver.run(nodes)

    assert results["good"].ok is True
    assert results["good"].attempts == 1
    assert results["bad"].ok is False
    assert results["bad"].attempts == 2
