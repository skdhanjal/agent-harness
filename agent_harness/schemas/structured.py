"""Structured Outputs: turn raw LLM text into a validated, typed contract."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ValidationError


class ChatClient(Protocol):
    """Anything with this method shape works here — OpenAI, a mock, etc."""

    def create_completion(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]]
    ) -> str: ...


class StructuredOutputError(RuntimeError):
    """Raised when a response still fails validation after all repairs."""


def _strip_code_fences(text: str) -> str:
    """Models often wrap JSON in ```json ... ``` even when told not to."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped.removeprefix("json").strip()
    return stripped


def generate_structured[T: BaseModel](
    client: ChatClient,
    schema: type[T],
    task_prompt: str,
    max_repairs: int = 3,
    tools: list[dict[str, object]] | None = None,
) -> T:
    """Call the model and return a validated instance of `schema`.

    Raises StructuredOutputError if the model can't produce valid output
    within `max_repairs` attempts.
    """
    tools = tools if tools is not None else []
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You must respond with a single JSON object and nothing else "
                "— no markdown fences, no commentary. It must validate "
                f"against this JSON schema:\n{schema_json}"
            ),
        },
        {"role": "user", "content": task_prompt},
    ]

    last_error: ValidationError | None = None
    for _ in range(max_repairs + 1):
        raw = client.create_completion(messages, tools)
        cleaned = _strip_code_fences(raw)
        try:
            return schema.model_validate_json(cleaned)
        except ValidationError as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That failed schema validation:\n{e}\n"
                        "Return corrected JSON only, no prose, no markdown fences."
                    ),
                }
            )

    raise StructuredOutputError(
        f"Failed to produce valid '{schema.__name__}' after {max_repairs} repairs: {last_error}"
    )
