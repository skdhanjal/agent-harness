"""Token counting: the actual currency the context budget is measured in."""

from __future__ import annotations

import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens the way a real model would, not by characters or words."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))
