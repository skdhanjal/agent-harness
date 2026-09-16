"""Verification tests for LLM-driven task decomposition
(agent_harness/orchestration/decomposition.py).

Proves: a valid decomposition becomes matching DagNodes, an empty or
cyclic decomposition is rejected and retried rather than handed to
SubAgentSpawner, a decomposition with any depends_on at all is rejected
too (SubAgentSpawner.run_dag can't honor ordering, even acyclic ordering),
and repeated invalid attempts eventually raise instead of retrying forever.
"""

import json

import pytest

from agent_harness.orchestration.decomposition import TaskDecompositionError, decompose_task


class FakeChatClient:
    """Returns a scripted sequence of responses, one per call. A dict
    response is JSON-encoded; a str response (e.g. malformed JSON) is
    returned as-is, to script a schema-repair-loop failure too.
    """

    def __init__(self, responses: list[dict[str, object] | str]) -> None:
        self._responses = responses
        self.call_count = 0

    def create_completion(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]]
    ) -> str:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response if isinstance(response, str) else json.dumps(response)


def _decomposition(*subtasks: dict[str, object]) -> dict[str, object]:
    return {"subtasks": list(subtasks)}


def _subtask(
    id_: str, task: str = "do it", difficulty: int = 0, depends_on: list[str] | None = None
) -> dict[str, object]:
    return {"id": id_, "task": task, "difficulty": difficulty, "depends_on": depends_on or []}


def test_valid_decomposition_becomes_matching_dag_nodes() -> None:
    client = FakeChatClient(
        [_decomposition(_subtask("a", "write notes on a", 0), _subtask("b", "write notes on b", 1))]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a", "b"]
    assert [n.task for n in nodes] == ["write notes on a", "write notes on b"]
    assert [n.difficulty for n in nodes] == [0, 1]
    assert client.call_count == 1


def test_empty_decomposition_is_retried_then_succeeds() -> None:
    client = FakeChatClient(
        [
            _decomposition(),
            _decomposition(_subtask("a")),
        ]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a"]
    assert client.call_count == 2


def test_cyclic_dependencies_are_rejected_then_retried() -> None:
    client = FakeChatClient(
        [
            _decomposition(
                _subtask("a", depends_on=["b"]),
                _subtask("b", depends_on=["a"]),
            ),
            _decomposition(_subtask("a"), _subtask("b")),
        ]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a", "b"]
    assert client.call_count == 2


def test_acyclic_dependency_is_still_rejected() -> None:
    """A single, acyclic depends_on edge is still invalid -- SubAgentSpawner
    always runs nodes in parallel, so cycle-freedom alone isn't enough.
    """
    client = FakeChatClient(
        [
            _decomposition(_subtask("a"), _subtask("b", depends_on=["a"])),
            _decomposition(_subtask("a"), _subtask("b")),
        ]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a", "b"]
    assert client.call_count == 2


def test_duplicate_subtask_ids_are_rejected_then_retried() -> None:
    client = FakeChatClient(
        [
            _decomposition(_subtask("a"), _subtask("a")),
            _decomposition(_subtask("a"), _subtask("b")),
        ]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a", "b"]
    assert client.call_count == 2


def test_depends_on_unknown_id_is_rejected_then_retried() -> None:
    client = FakeChatClient(
        [
            _decomposition(_subtask("a", depends_on=["ghost"])),
            _decomposition(_subtask("a")),
        ]
    )

    nodes = decompose_task("split this", client)

    assert [n.id for n in nodes] == ["a"]
    assert client.call_count == 2


def test_malformed_json_exhausting_the_schema_repair_loop_is_retried_too() -> None:
    """generate_structured's own repair loop (default max_repairs=3, so 4
    raw calls) handles malformed JSON internally; only once *that* is
    exhausted does decompose_task see a StructuredOutputError, which it
    retries exactly like a business-rule validation failure.
    """
    client = FakeChatClient(["not json"] * 4 + [_decomposition(_subtask("a"))])

    nodes = decompose_task("split this", client, max_attempts=2)

    assert [n.id for n in nodes] == ["a"]
    assert client.call_count == 5


def test_raises_after_exhausting_attempts_on_persistently_invalid_output() -> None:
    client = FakeChatClient([_decomposition()] * 10)

    with pytest.raises(TaskDecompositionError, match="at least one subtask"):
        decompose_task("split this", client, max_attempts=3)

    assert client.call_count == 3
