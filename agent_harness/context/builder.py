"""Context Construction: budget the context window like the scarce resource it is.

Concatenating everything until you hit a wall works until it doesn't --
cost explodes, and models attend worse to information buried in the
middle of a long context. This module actively curates what goes in,
by priority, instead of just appending until something breaks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ContextBlock:
    name: str
    content: str
    priority: int  # lower = more important, kept first
    tokens: int


class ContextBuilder:
    """Assembles context from prioritized blocks, truncating lowest-priority first.

    Truncation is whole-block: a block either fits entirely or is replaced
    by a short placeholder. No mid-sentence character-level cuts.
    """

    def __init__(self, token_budget: int, count_tokens: Callable[[str], int]) -> None:
        self.token_budget = token_budget
        self._count_tokens = count_tokens
        self._blocks: list[ContextBlock] = []

    def add(self, name: str, content: str, priority: int) -> None:
        self._blocks.append(
            ContextBlock(
                name=name,
                content=content,
                priority=priority,
                tokens=self._count_tokens(content),
            )
        )

    def build(self) -> str:
        ordered = sorted(self._blocks, key=lambda b: b.priority)
        kept: list[ContextBlock] = []
        used = 0
        for block in ordered:
            if used + block.tokens <= self.token_budget:
                kept.append(block)
                used += block.tokens
            else:
                kept.append(
                    ContextBlock(
                        name=block.name,
                        content=f"[[{block.name} truncated: over budget]]",
                        priority=block.priority,
                        tokens=0,
                    )
                )
        return "\n\n".join(f"### {b.name}\n{b.content}" for b in kept)
