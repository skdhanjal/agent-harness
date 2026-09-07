"""Verification test for Module 1: Structured Outputs.

Proves the repair loop actually repairs, and fails safely (not silently,
not infinitely) when the model can't be salvaged. Uses a fake client —
no real API calls, so this runs the same in CI as it does locally.
"""

import pytest
from pydantic import BaseModel

from agent_harness.schemas.structured import StructuredOutputError, generate_structured


class Point(BaseModel):
    x: int
    y: int


class FakeChatClient:
    """Returns a scripted sequence of responses, one per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def test_repair_loop_recovers_after_invalid_attempts() -> None:
    client = FakeChatClient(
        responses=[
            '{"x": "not_a_number", "y": 2}',  # invalid type
            "not even json",  # invalid JSON
            '{"x": 1, "y": 2}',  # valid
        ]
    )

    result = generate_structured(client, Point, "Give me a point.", max_repairs=3)

    assert result == Point(x=1, y=2)
    assert client.call_count == 3


def test_raises_after_exhausting_repairs() -> None:
    client = FakeChatClient(responses=["not json"] * 10)

    with pytest.raises(StructuredOutputError):
        generate_structured(client, Point, "Give me a point.", max_repairs=2)

    assert client.call_count == 3  # initial attempt + 2 repairs


def test_strips_markdown_code_fences() -> None:
    client = FakeChatClient(responses=['```json\n{"x": 5, "y": 6}\n```'])

    result = generate_structured(client, Point, "Give me a point.", max_repairs=0)

    assert result == Point(x=5, y=6)
