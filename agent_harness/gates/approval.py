"""Permission & Approval Gates: not every action deserves the same trust.

Reading a file and deleting a database shouldn't go through the same
unsupervised path. This inserts a deterministic pause point -- a human
(or a stricter policy) must approve high-risk actions before they run.
The harness sets the threshold; a human makes the actual call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from agent_harness.gates import pending_store


class RiskTier(IntEnum):
    """IntEnum so tiers compare naturally: LOW < MEDIUM < HIGH."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class PendingAction:
    id: str
    tool_name: str
    args: dict[str, object]
    risk: RiskTier
    run_id: str


class ApprovalGate:
    """Auto-approves below a threshold; everything at/above it needs a human.

    Decisions are persisted by `action.id` (the provider's own tool_call_id)
    so a resumed run replays an already-made decision instead of asking
    again, and a crash mid-approval still leaves evidence of what was
    pending -- see docs/agent_harness_roadmap.md's Module 8 section.
    """

    def __init__(
        self,
        require_approval_at: RiskTier = RiskTier.HIGH,
        approver: Callable[[PendingAction], bool] | None = None,
        pending_dir: str = "./pending_actions",
    ) -> None:
        self.require_approval_at = require_approval_at
        self._approver = approver or self._cli_approver
        self.pending_dir = pending_dir

    def check(self, action: PendingAction) -> bool:
        if action.risk < self.require_approval_at:
            return True

        status = pending_store.load_status(action.run_id, action.id, directory=self.pending_dir)
        if status == "approved":
            return True
        if status == "denied":
            return False

        pending_store.mark_pending(
            action.run_id, action.id, action.tool_name, action.args, directory=self.pending_dir
        )
        approved = self._approver(action)
        pending_store.resolve(action.run_id, action.id, approved, directory=self.pending_dir)
        return approved

    @staticmethod
    def _cli_approver(action: PendingAction) -> bool:
        response = input(
            f"[APPROVAL REQUIRED] {action.tool_name}({action.args}) "
            f"risk={action.risk.name}. Approve? [y/N]: "
        )
        return response.strip().lower() == "y"
