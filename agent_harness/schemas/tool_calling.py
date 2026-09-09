"""Native tool-calling contract: the provider's own structured tool_calls,
not a hand-rolled JSON action schema (contrast with Module 1's
`generate_structured`, which asks the model to produce text matching a
schema and repairs it on failure after the fact).

Tool selection here is validated by the provider *at generation time* --
the model returns either one or more tool calls (each already parsed,
named, and id-tagged) or a plain-text final answer, never an ambiguous
blob of both squeezed into one JSON object.
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


class ToolCallingChatClient(Protocol):
    """Anything with this method shape works here — OpenAI, a mock, etc."""

    def create_action_turn(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn: ...
