"""Verification tests for wiring DurableStateStore into the tool registry.

Proves: remember_fact/recall_fact/list_facts work through the same
validate-execute-never-raise path as any other tool, and facts genuinely
survive a fresh DurableStateStore pointed at the same file -- the whole
point of cross-run memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.memory.durable_store import DurableStateStore
from agent_harness.memory.tools import register_memory_tools
from agent_harness.tools.registry import ToolRegistry


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "facts.json"


def _registry(store: DurableStateStore) -> ToolRegistry:
    reg = ToolRegistry()
    register_memory_tools(reg, store)
    return reg


def test_remember_then_recall_round_trips(store_path: Path) -> None:
    registry = _registry(DurableStateStore(store_path))

    remember = registry.execute("remember_fact", {"key": "favorite_color", "value": "blue"})
    assert remember.ok is True

    recall = registry.execute("recall_fact", {"key": "favorite_color"})
    assert recall.ok is True
    assert recall.result == "blue"


def test_recall_missing_key_returns_sentinel_not_error(store_path: Path) -> None:
    registry = _registry(DurableStateStore(store_path))

    result = registry.execute("recall_fact", {"key": "does_not_exist"})

    assert result.ok is True
    assert result.result == "(no fact stored for this key)"


def test_list_facts_returns_remembered_keys(store_path: Path) -> None:
    registry = _registry(DurableStateStore(store_path))
    registry.execute("remember_fact", {"key": "a", "value": "1"})
    registry.execute("remember_fact", {"key": "b", "value": "2"})

    result = registry.execute("list_facts", {})

    assert result.ok is True
    assert sorted(result.result) == ["a", "b"]


def test_facts_survive_a_fresh_registry_and_store(store_path: Path) -> None:
    """The point of this module: a fact from one run is visible in the next,
    a fresh process with only the shared file in common with the last one.
    """
    first_run = _registry(DurableStateStore(store_path))
    first_run.execute("remember_fact", {"key": "user_name", "value": "Sam"})

    second_run = _registry(DurableStateStore(store_path))
    result = second_run.execute("recall_fact", {"key": "user_name"})

    assert result.ok is True
    assert result.result == "Sam"
