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


class RiskTier(IntEnum):
    """IntEnum so tiers compare naturally: LOW < MEDIUM < HIGH."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class PendingAction:
    tool_name: str
    args: dict[str, object]
    risk: RiskTier
    run_id: str


class ApprovalGate:
    """Auto-approves below a threshold; everything at/above it needs a human."""

    def __init__(
        self,
        require_approval_at: RiskTier = RiskTier.HIGH,
        approver: Callable[[PendingAction], bool] | None = None,
    ) -> None:
        self.require_approval_at = require_approval_at
        self._approver = approver or self._cli_approver

    def check(self, action: PendingAction) -> bool:
        if action.risk < self.require_approval_at:
            return True
        return self._approver(action)

    @staticmethod
    def _cli_approver(action: PendingAction) -> bool:
        response = input(
            f"[APPROVAL REQUIRED] {action.tool_name}({action.args}) "
            f"risk={action.risk.name}. Approve? [y/N]: "
        )
        return response.strip().lower() == "y"
