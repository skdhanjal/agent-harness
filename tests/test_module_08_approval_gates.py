"""Verification tests for Module 8: Permission & Approval Gates.

Proves: risk below the threshold never touches the approver at all,
risk at/above it defers entirely to the approver's decision (approve
or deny), and the threshold itself is configurable.
"""

from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier


def make_action(risk: RiskTier) -> PendingAction:
    return PendingAction(
        tool_name="delete_file", args={"path": "/tmp/x"}, risk=risk, run_id="run-1"
    )


def test_low_risk_never_invokes_the_approver() -> None:
    def approver_that_should_never_run(action: PendingAction) -> bool:
        raise AssertionError("approver should not be called for low risk")

    gate = ApprovalGate(approver=approver_that_should_never_run)

    assert gate.check(make_action(RiskTier.LOW)) is True


def test_high_risk_respects_approval() -> None:
    gate = ApprovalGate(approver=lambda action: True)

    assert gate.check(make_action(RiskTier.HIGH)) is True


def test_high_risk_respects_denial() -> None:
    gate = ApprovalGate(approver=lambda action: False)

    assert gate.check(make_action(RiskTier.HIGH)) is False


def test_medium_risk_auto_approved_under_default_threshold() -> None:
    def approver_that_should_never_run(action: PendingAction) -> bool:
        raise AssertionError("default threshold is HIGH; MEDIUM should auto-approve")

    gate = ApprovalGate(approver=approver_that_should_never_run)

    assert gate.check(make_action(RiskTier.MEDIUM)) is True


def test_threshold_can_be_lowered_to_require_approval_at_medium() -> None:
    gate = ApprovalGate(require_approval_at=RiskTier.MEDIUM, approver=lambda action: False)

    assert gate.check(make_action(RiskTier.MEDIUM)) is False
