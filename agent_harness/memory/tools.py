"""Exposes DurableStateStore to the model as tools, the same way filesystem
access is exposed -- the model decides when a fact is worth persisting or
recalling, rather than the harness guessing.
"""

from __future__ import annotations

from agent_harness.memory.durable_store import DurableStateStore
from agent_harness.tools.registry import ToolRegistry


def register_memory_tools(registry: ToolRegistry, store: DurableStateStore) -> None:
    @registry.tool
    def remember_fact(key: str, value: str) -> str:
        """Persist a fact as a key-value pair that survives across separate
        agent runs, not just this conversation. Overwrites any existing
        value stored under the same key.
        """
        store.set(key, value)
        return f"remembered '{key}'"

    @registry.tool
    def recall_fact(key: str) -> str:
        """Recall a fact previously saved with remember_fact. Returns
        '(no fact stored for this key)' if nothing was ever saved under it.
        """
        value = store.get(key)
        return str(value) if value is not None else "(no fact stored for this key)"

    @registry.tool
    def list_facts() -> list[str]:
        """List the keys of all facts currently remembered, so you can see
        what's already known before recalling or overwriting a fact.
        """
        return store.keys()
