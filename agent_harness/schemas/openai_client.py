"""Thin adapter: makes the OpenAI SDK match the harness's ChatClient shape.

The rest of the harness never imports `openai` directly — only this file
does. If we ever route across providers (Module 10), this is the only
place that changes.
"""

from __future__ import annotations

import json
import os
from typing import cast

from dotenv import load_dotenv
from openai import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from agent_harness.persistence.retry import FatalError
from agent_harness.schemas.tool_calling import ChatTurn, ToolCallRequest

load_dotenv()

# Client errors that retrying can never fix -- bad credentials, malformed
# requests, wrong model/permissions. Re-raised as FatalError so with_backoff
# fails fast instead of burning 5 retries on something that won't change.
# Rate limits, server errors, and connection issues are left alone: those
# are exactly what backoff is for.
_NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
)

# USD per 1M tokens, (input, output) -- OpenAI's published prices at write
# time. Unrecognized models fall back to gpt-4o-mini's rate rather than
# raising, since a cost estimate is better than crashing a live run over it.
_PRICE_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class OpenAIChatClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def _cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        input_price, output_price = _PRICE_PER_MILLION_TOKENS.get(
            self._model, _PRICE_PER_MILLION_TOKENS["gpt-4o-mini"]
        )
        return (tokens_in * input_price + tokens_out * output_price) / 1_000_000

    def create_completion(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]]
    ) -> str:
        # tools are described as prompt text, not passed to the API's native
        # tool-calling -- mixing that with forced JSON mode corrupts output.
        prepared = list(messages)
        if tools:
            tool_text = (
                "Available tools -- to use one, set tool_name/tool_args in "
                "your JSON response accordingly:\n" + json.dumps(tools, indent=2)
            )
            prepared.insert(0, {"role": "system", "content": tool_text})

        typed_messages = cast(list[ChatCompletionMessageParam], prepared)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=typed_messages,
                response_format={"type": "json_object"},
            )
        except _NON_RETRYABLE_ERRORS as exc:
            raise FatalError(str(exc)) from exc
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Model returned empty content")
        return content

    def create_action_turn(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        """One native tool-calling turn: no forced JSON mode, so the model
        can reply with tool calls or plain text.
        """
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        try:
            if tools:
                typed_tools = cast(list[ChatCompletionToolUnionParam], tools)
                response = self._client.chat.completions.create(
                    model=self._model, messages=typed_messages, tools=typed_tools
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model, messages=typed_messages
                )
        except _NON_RETRYABLE_ERRORS as exc:
            raise FatalError(str(exc)) from exc

        message = response.choices[0].message
        # only "function" tools are ever registered, so narrow to that variant
        tool_calls = [
            ToolCallRequest(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments) if call.function.arguments else {},
            )
            for call in message.tool_calls or []
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        ]
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        return ChatTurn(
            content=message.content,
            tool_calls=tool_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost_usd(tokens_in, tokens_out),
        )
