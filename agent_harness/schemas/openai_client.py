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
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()


class OpenAIChatClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def create_completion(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]]
    ) -> str:
        # Tool schemas are surfaced as prompt text, not the native `tools=`
        # kwarg: handing them to the API's real function-calling machinery
        # makes the model respond in that format (e.g. tool_name comes back
        # prefixed "functions.", or action_type drifts to an invalid value
        # like "multi_tool_use.parallel") instead of the harness's own
        # AgentAction JSON contract, and it also biases the model to keep
        # invoking a tool even after the task is already done.
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
