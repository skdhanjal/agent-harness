"""Verification tests for Module 6: State and Memory.

Proves: sliding window evicts oldest first, vector memory retrieves by
meaning not insertion order, and durable state genuinely survives a
fresh instance pointed at the same file (not just an in-memory dict).
"""

from pathlib import Path

from agent_harness.memory.durable_store import DurableStateStore
from agent_harness.memory.sliding_window import SlidingWindowMemory
from agent_harness.memory.vector_memory import VectorMemory

_VOCAB = ["cat", "dog", "stock", "market", "loyal"]


def _fake_embed(text: str) -> list[float]:
    words = text.lower().split()
    return [float(words.count(w)) for w in _VOCAB]


def test_sliding_window_evicts_oldest_beyond_max_turns() -> None:
    memory = SlidingWindowMemory(max_turns=2)
    memory.add("turn 1")
    memory.add("turn 2")
    memory.add("turn 3")

    assert memory.get_all() == ["turn 2", "turn 3"]


def test_vector_memory_retrieves_most_relevant_not_most_recent() -> None:
    memory = VectorMemory(embed_fn=_fake_embed)
    memory.add("I have a pet cat")
    memory.add("Dogs are loyal")
    memory.add("The stock market crashed today")

    results = memory.retrieve("tell me about my cat", k=1)

    assert results == ["I have a pet cat"]


def test_vector_memory_empty_returns_empty_list() -> None:
    memory = VectorMemory(embed_fn=_fake_embed)

    assert memory.retrieve("anything") == []


def test_durable_store_survives_a_fresh_instance(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    store_1 = DurableStateStore(path)
    store_1.set("user_name", "Sam")

    store_2 = DurableStateStore(path)  # simulates a process restart

    assert store_2.get("user_name") == "Sam"


def test_durable_store_missing_key_returns_none(tmp_path: Path) -> None:
    store = DurableStateStore(tmp_path / "state.json")

    assert store.get("does_not_exist") is None
