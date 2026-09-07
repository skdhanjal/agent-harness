"""Pydantic contracts for what the agent is allowed to say.

Every response the model produces must validate against one of these
before any other module in the harness is allowed to touch it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class AgentAction(BaseModel):
    """A single decision the agent makes at one step of its loop."""

    reasoning: str = Field(..., description="Brief reasoning for this action.")
    action_type: ActionType
    tool_name: str | None = Field(
        default=None, description="Required when action_type is 'tool_call'."
    )
    tool_args: dict[str, object] = Field(default_factory=dict)
    final_answer: str | None = Field(
        default=None, description="Required when action_type is 'final_answer'."
    )
