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
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from agent_harness.schemas.tool_calling import ChatTurn, ToolCallRequest

load_dotenv()


class OpenAIChatClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=typed_messages,
            response_format={"type": "json_object"},
        )
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
        if tools:
            typed_tools = cast(list[ChatCompletionToolUnionParam], tools)
            response = self._client.chat.completions.create(
                model=self._model, messages=typed_messages, tools=typed_tools
            )
        else:
            response = self._client.chat.completions.create(
                model=self._model, messages=typed_messages
            )

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
        return ChatTurn(content=message.content, tool_calls=tool_calls)
