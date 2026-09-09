"""Short-term memory: the last N turns, verbatim. Oldest evicted first."""

from __future__ import annotations

from collections import deque


class SlidingWindowMemory:
    def __init__(self, max_turns: int) -> None:
        self._turns: deque[str] = deque(maxlen=max_turns)

    def add(self, turn: str) -> None:
        self._turns.append(turn)

    def get_all(self) -> list[str]:
        return list(self._turns)
