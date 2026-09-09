"""Adaptive Orchestration -- model routing half.

Not every task deserves the most expensive model. Route by estimated
difficulty, and escalate specifically when something has already failed,
rather than defaulting everything to the top tier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTier:
    name: str
    cost_per_1k: float
    capability_rank: int  # higher = more capable


class ModelRouter:
    def __init__(self, tiers: list[ModelTier]) -> None:
        self._tiers = sorted(tiers, key=lambda t: t.capability_rank)

    def choose(self, estimated_difficulty: int, prior_failures: int = 0) -> ModelTier:
        index = min(estimated_difficulty + prior_failures, len(self._tiers) - 1)
        return self._tiers[index]

    def escalate(self, current: ModelTier) -> ModelTier:
        index = self._tiers.index(current)
        return self._tiers[min(index + 1, len(self._tiers) - 1)]
