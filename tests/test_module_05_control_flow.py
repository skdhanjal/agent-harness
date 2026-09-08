"""Verification tests for Module 5: Control-Flow Ownership.

Proves: legal transitions succeed, illegal ones are rejected without
changing state, the step-count guard trips into FAILED regardless of
what's requested, and terminal states (DONE/FAILED) refuse everything
afterward -- the model can't talk its way past a tripped guard.
"""

import pytest

from agent_harness.control_flow.state_machine import (
    AgentFSM,
    IllegalTransitionError,
    Phase,
    StepLimitExceededError,
)


def test_legal_transition_sequence_succeeds() -> None:
    fsm = AgentFSM()

    fsm.step(Phase.EXECUTE)
    fsm.step(Phase.REVIEW)
    fsm.step(Phase.DONE)

    assert fsm.phase == Phase.DONE

    with pytest.raises(IllegalTransitionError):
        fsm.step(Phase.EXECUTE)  # DONE is terminal, nothing moves it again


def test_illegal_transition_raises_and_does_not_change_phase() -> None:
    fsm = AgentFSM()

    with pytest.raises(IllegalTransitionError):
        fsm.step(Phase.DONE)  # PLAN -> DONE skips EXECUTE and REVIEW entirely

    assert fsm.phase == Phase.PLAN  # rejected transition leaves state untouched


def test_execute_cannot_skip_review_straight_to_done() -> None:
    fsm = AgentFSM()
    fsm.step(Phase.EXECUTE)

    with pytest.raises(IllegalTransitionError):
        fsm.step(Phase.DONE)


def test_step_limit_guard_trips_and_lands_in_failed() -> None:
    fsm = AgentFSM(max_steps=3)
    fsm.step(Phase.EXECUTE)  # step 1
    fsm.step(Phase.EXECUTE)  # step 2, legal self-loop
    fsm.step(Phase.EXECUTE)  # step 3

    with pytest.raises(StepLimitExceededError):
        fsm.step(Phase.EXECUTE)  # step 4 -- exceeds max_steps

    assert fsm.phase == Phase.FAILED


def test_model_cannot_talk_its_way_past_a_tripped_guard() -> None:
    fsm = AgentFSM(max_steps=1)
    fsm.step(Phase.EXECUTE)  # step 1, within budget

    with pytest.raises(StepLimitExceededError):
        fsm.step(Phase.EXECUTE)  # step 2, exceeds guard -> FAILED

    # FAILED is terminal -- even a "legal-looking" request is refused
    with pytest.raises(IllegalTransitionError):
        fsm.step(Phase.EXECUTE)
