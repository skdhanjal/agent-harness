"""Thin adapter: makes the OpenAI SDK match the harness's ChatClient shape.

The rest of the harness never imports `openai` directly — only this file
does. If we ever route across providers (Module 10), this is the only
place that changes.
"""

from __future__ import annotations

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

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=typed_messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Model returned empty content")
        return content
