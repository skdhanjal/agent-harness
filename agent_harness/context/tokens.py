"""Token counting: the actual currency the context budget is measured in."""

from __future__ import annotations

import json
from collections.abc import Callable

import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens the way a real model would, not by characters or words."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_messages_tokens(
    messages: list[dict[str, object]], count_tokens: Callable[[str], int]
) -> int:
    """Token count of a whole message list -- structure like tool_calls
    costs tokens too, not just `content`."""
    return sum(count_tokens(json.dumps(message)) for message in messages)
