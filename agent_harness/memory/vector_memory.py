"""Long-term memory: retrieve by meaning (embedding similarity), not recency.

embed_fn is injected -- same pattern as ChatClient (Module 1) and
count_tokens (Module 4) -- so tests never depend on a real embeddings API.
"""

from __future__ import annotations

import math
from collections.abc import Callable


class VectorMemory:
    def __init__(self, embed_fn: Callable[[str], list[float]]) -> None:
        self._embed_fn = embed_fn
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, text: str) -> None:
        self._texts.append(text)
        self._vectors.append(self._embed_fn(text))

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        if not self._vectors:
            return []
        query_vec = self._embed_fn(query)
        scored = [
            (self._cosine_similarity(query_vec, v), text)
            for v, text in zip(self._vectors, self._texts, strict=True)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [text for _, text in scored[:k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
