"""Native tool-calling contract: the provider validates tool selection at
generation time, instead of the model writing JSON against a hand-rolled
schema (contrast with Module 1's `generate_structured`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ToolCallRequest:
    """One provider-issued tool call: already parsed, never raw JSON text."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatTurn:
    """What the model did this turn: some tool calls, a final answer, or both."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class ToolCallingChatClient(Protocol):
    """Anything with this method shape works here — OpenAI, a mock, etc."""

    def create_action_turn(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn: ...
