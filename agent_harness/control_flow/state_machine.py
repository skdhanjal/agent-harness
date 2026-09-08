"""Control-Flow Ownership: the harness decides what happens next, not the model.

An agent that loops "call LLM -> maybe act -> repeat" until it *feels*
done is nondeterministic and undebuggable. This module makes the legal
states and transitions explicit and enforced in code -- the model only
proposes a transition; the FSM decides whether it's actually allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    PLAN = auto()
    EXECUTE = auto()
    REVIEW = auto()
    DONE = auto()
    FAILED = auto()


class IllegalTransitionError(ValueError):
    """Raised when a requested phase transition isn't in the allowed graph,
    or when the FSM is already in a terminal state."""


class StepLimitExceededError(RuntimeError):
    """Raised when the FSM would take more steps than its hard guard allows."""


_TERMINAL_PHASES = frozenset({Phase.DONE, Phase.FAILED})


def _default_transitions() -> dict[Phase, set[Phase]]:
    return {
        Phase.PLAN: {Phase.EXECUTE},
        Phase.EXECUTE: {Phase.REVIEW, Phase.EXECUTE},
        Phase.REVIEW: {Phase.EXECUTE, Phase.DONE, Phase.FAILED},
    }


@dataclass
class AgentFSM:
    """Deterministic phase machine. The model proposes; this approves or rejects."""

    phase: Phase = Phase.PLAN
    step_count: int = 0
    max_steps: int = 25
    transitions: dict[Phase, set[Phase]] = field(default_factory=_default_transitions)

    def step(self, requested_phase: Phase) -> None:
        if self.phase in _TERMINAL_PHASES:
            raise IllegalTransitionError(
                f"FSM is in terminal state {self.phase.name}; no further transitions allowed"
            )

        self.step_count += 1

        if self.step_count > self.max_steps:
            self.phase = Phase.FAILED
            raise StepLimitExceededError(
                f"max_steps={self.max_steps} exceeded -- control-flow guard tripped"
            )

        allowed = self.transitions.get(self.phase, set())

        if requested_phase not in allowed:
            raise IllegalTransitionError(
                f"Illegal transition {self.phase.name} -> {requested_phase.name}"
            )

        self.phase = requested_phase
