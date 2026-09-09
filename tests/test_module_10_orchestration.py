"""Verification tests for Module 10: Adaptive Orchestration.

Proves: the router picks higher tiers for harder tasks, never exceeds
the top tier regardless of how bad things get, and escalate() moves up
one tier at a time, capping at the top. Proves the spawner runs all
nodes to completion while never exceeding max_workers concurrently --
the real correctness property, not just "it eventually finishes."
"""

import threading
import time

from agent_harness.orchestration.router import ModelRouter, ModelTier
from agent_harness.orchestration.spawner import DagNode, SubAgentSpawner


def test_router_chooses_higher_tier_for_higher_difficulty() -> None:
    router = ModelRouter(
        [
            ModelTier("fast", cost_per_1k=0.001, capability_rank=0),
            ModelTier("balanced", cost_per_1k=0.01, capability_rank=1),
            ModelTier("frontier", cost_per_1k=0.1, capability_rank=2),
        ]
    )

    assert router.choose(estimated_difficulty=0).name == "fast"
    assert router.choose(estimated_difficulty=1).name == "balanced"
    assert router.choose(estimated_difficulty=2).name == "frontier"


def test_router_never_exceeds_top_tier_even_with_many_failures() -> None:
    router = ModelRouter(
        [
            ModelTier("fast", cost_per_1k=0.001, capability_rank=0),
            ModelTier("frontier", cost_per_1k=0.1, capability_rank=1),
        ]
    )

    result = router.choose(estimated_difficulty=5, prior_failures=10)

    assert result.name == "frontier"


def test_router_escalate_moves_to_next_tier() -> None:
    fast = ModelTier("fast", cost_per_1k=0.001, capability_rank=0)
    frontier = ModelTier("frontier", cost_per_1k=0.1, capability_rank=1)
    router = ModelRouter([fast, frontier])

    assert router.escalate(fast).name == "frontier"


def test_router_escalate_caps_at_top_tier() -> None:
    fast = ModelTier("fast", cost_per_1k=0.001, capability_rank=0)
    frontier = ModelTier("frontier", cost_per_1k=0.1, capability_rank=1)
    router = ModelRouter([fast, frontier])

    assert router.escalate(frontier).name == "frontier"


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class _TrackingHarness:
    def __init__(self, tracker: _ConcurrencyTracker) -> None:
        self._tracker = tracker

    def run(self, task: str) -> str:
        self._tracker.enter()
        time.sleep(0.05)
        self._tracker.exit()
        return f"done:{task}"


def test_spawner_completes_all_nodes_and_respects_max_workers() -> None:
    tracker = _ConcurrencyTracker()
    spawner = SubAgentSpawner(harness_factory=lambda: _TrackingHarness(tracker), max_workers=2)
    nodes = [DagNode(id=f"n{i}", task=f"task-{i}") for i in range(5)]

    results = spawner.run_dag(nodes)

    assert len(results) == 5
    assert all(results[f"n{i}"] == f"done:task-{i}" for i in range(5))
    assert tracker.peak <= 2
