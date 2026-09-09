"""LLM-as-a-Judge: turn "does this look right" into a repeatable score.

Reuses Module 1's ChatClient protocol -- tests inject a scripted fake
client, so scoring logic is verified without a real API call.
"""

from __future__ import annotations

from agent_harness.schemas.structured import ChatClient


class LLMJudge:
    def __init__(self, client: ChatClient, rubric: str) -> None:
        self._client = client
        self._rubric = rubric

    def score(self, trajectory: str) -> int:
        messages = [
            {
                "role": "system",
                "content": (
                    f"Score the trajectory 0-10 against this rubric:\n{self._rubric}\n"
                    "Respond with only the integer score, nothing else."
                ),
            },
            {"role": "user", "content": trajectory},
        ]
        raw = self._client.create_completion(messages)
        return int(raw.strip())
