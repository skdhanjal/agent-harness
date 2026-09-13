"""Verification tests for Module 8: Permission & Approval Gates.

Proves: risk below the threshold never touches the approver at all,
risk at/above it defers entirely to the approver's decision (approve
or deny), the threshold itself is configurable, and a decision once
made is durable -- persisted before the approver even runs, and never
re-asked once it resolves.
"""

from pathlib import Path

from agent_harness.gates import pending_store
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier


def make_action(risk: RiskTier, action_id: str = "call-1") -> PendingAction:
    return PendingAction(
        id=action_id, tool_name="delete_file", args={"path": "/tmp/x"}, risk=risk, run_id="run-1"
    )


def test_low_risk_never_invokes_the_approver() -> None:
    def approver_that_should_never_run(action: PendingAction) -> bool:
        raise AssertionError("approver should not be called for low risk")

    gate = ApprovalGate(approver=approver_that_should_never_run)

    assert gate.check(make_action(RiskTier.LOW)) is True


def test_high_risk_respects_approval(tmp_path: Path) -> None:
    gate = ApprovalGate(approver=lambda action: True, pending_dir=str(tmp_path))

    assert gate.check(make_action(RiskTier.HIGH)) is True


def test_high_risk_respects_denial(tmp_path: Path) -> None:
    gate = ApprovalGate(approver=lambda action: False, pending_dir=str(tmp_path))

    assert gate.check(make_action(RiskTier.HIGH)) is False


def test_medium_risk_auto_approved_under_default_threshold() -> None:
    def approver_that_should_never_run(action: PendingAction) -> bool:
        raise AssertionError("default threshold is HIGH; MEDIUM should auto-approve")

    gate = ApprovalGate(approver=approver_that_should_never_run)

    assert gate.check(make_action(RiskTier.MEDIUM)) is True


def test_threshold_can_be_lowered_to_require_approval_at_medium(tmp_path: Path) -> None:
    gate = ApprovalGate(
        require_approval_at=RiskTier.MEDIUM,
        approver=lambda action: False,
        pending_dir=str(tmp_path),
    )

    assert gate.check(make_action(RiskTier.MEDIUM)) is False


def test_pending_record_is_written_before_the_approver_runs(tmp_path: Path) -> None:
    """Crash durability means evidence survives even if the process dies
    mid-approval -- so the record must exist on disk *before* the (possibly
    blocking, possibly crashing) approver call, not after."""
    action = make_action(RiskTier.HIGH)

    def approver_checks_pending_first(pending: PendingAction) -> bool:
        assert (
            pending_store.load_status(pending.run_id, pending.id, directory=str(tmp_path))
            == "pending"
        )
        return True

    gate = ApprovalGate(approver=approver_checks_pending_first, pending_dir=str(tmp_path))

    assert gate.check(action) is True


def test_resumed_gate_replays_an_already_approved_decision_without_reasking(tmp_path: Path) -> None:
    action = make_action(RiskTier.HIGH)
    first_gate = ApprovalGate(approver=lambda a: True, pending_dir=str(tmp_path))
    assert first_gate.check(action) is True

    def approver_that_should_never_run(a: PendingAction) -> bool:
        raise AssertionError("decision already resolved; approver should not run again")

    resumed_gate = ApprovalGate(approver=approver_that_should_never_run, pending_dir=str(tmp_path))

    assert resumed_gate.check(action) is True


def test_resumed_gate_replays_an_already_denied_decision_without_reasking(tmp_path: Path) -> None:
    action = make_action(RiskTier.HIGH)
    first_gate = ApprovalGate(approver=lambda a: False, pending_dir=str(tmp_path))
    assert first_gate.check(action) is False

    def approver_that_should_never_run(a: PendingAction) -> bool:
        raise AssertionError("decision already resolved; approver should not run again")

    resumed_gate = ApprovalGate(approver=approver_that_should_never_run, pending_dir=str(tmp_path))

    assert resumed_gate.check(action) is False
