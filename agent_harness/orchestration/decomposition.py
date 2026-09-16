"""Adaptive Orchestration -- task decomposition: turn one task string into
a list[DagNode] for OrchestrationDriver/SubAgentSpawner, instead of a
caller having to hand-write the DAG.

Reuses Module 1's generate_structured/ChatClient rather than the harness's
native tool-calling client -- this is a single one-shot structured request,
not a tool-using agent turn.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.orchestration.spawner import DagNode
from agent_harness.schemas.structured import ChatClient, StructuredOutputError, generate_structured

_DECOMPOSE_PROMPT_TEMPLATE = (
    "Break the following task into 1-5 subtasks that can each run fully "
    "independently and in parallel -- no subtask's work may depend on "
    "another subtask's output or ordering, since they always execute "
    "concurrently. Leave depends_on empty on every subtask; it only exists "
    "for you to flag a dependency you can't avoid, which makes the whole "
    "decomposition invalid and forces a retry. If the task genuinely can't "
    "be split into independent pieces, return exactly one subtask covering "
    "the whole task. Give each subtask a short unique id, the concrete "
    "task text, and a difficulty estimate from 0 (simple) to 2 (hard)."
    "\n\nTask: {task}"
)


class SubtaskSpec(BaseModel):
    id: str
    task: str
    difficulty: int = Field(default=0, ge=0)
    depends_on: list[str] = Field(default_factory=list)


class TaskDecomposition(BaseModel):
    subtasks: list[SubtaskSpec]


class TaskDecompositionError(RuntimeError):
    """Raised when no valid decomposition was produced within max_attempts."""


def _has_cycle(subtasks: list[SubtaskSpec]) -> bool:
    """Kahn's algorithm: if a topological sort can't place every node,
    some subset of depends_on edges forms a cycle.
    """
    in_degree = {s.id: 0 for s in subtasks}
    dependents: dict[str, list[str]] = {s.id: [] for s in subtasks}
    for s in subtasks:
        for dep in s.depends_on:
            if dep in dependents:
                dependents[dep].append(s.id)
                in_degree[s.id] += 1

    queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for dependent in dependents[node_id]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    return visited != len(subtasks)


def _validate(subtasks: list[SubtaskSpec]) -> str | None:
    """Returns a description of what's wrong, or None if valid. Checked
    most-specific-diagnosis first; the last check is the one that actually
    matters given SubAgentSpawner.run_dag's real capability -- the earlier
    ones just produce a more informative message when that's the better
    diagnosis for the same underlying problem.
    """
    if not subtasks:
        return "decomposition must contain at least one subtask"

    ids = [s.id for s in subtasks]
    if len(set(ids)) != len(ids):
        return f"duplicate subtask ids: {ids}"

    id_set = set(ids)
    for s in subtasks:
        for dep in s.depends_on:
            if dep not in id_set:
                return f"subtask '{s.id}' has depends_on referencing unknown id '{dep}'"

    if _has_cycle(subtasks):
        return "dependency graph contains a cycle"

    if any(s.depends_on for s in subtasks):
        return (
            "SubAgentSpawner.run_dag has no dependency-edge resolution -- subtasks "
            "always run fully in parallel, so depends_on must be empty on every subtask"
        )

    return None


def decompose_task(task: str, client: ChatClient, max_attempts: int = 3) -> list[DagNode]:
    """Splits `task` into independent DagNodes via one or more structured-output
    calls, rejecting and retrying an empty, cyclic, or ordering-dependent
    decomposition rather than silently handing SubAgentSpawner something it
    can't honor.
    """
    base_prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(task=task)
    last_error: str | None = None

    for _ in range(max_attempts):
        prompt = (
            base_prompt
            if last_error is None
            else f"{base_prompt}\n\nYour previous attempt was invalid: {last_error}. Try again."
        )
        try:
            decomposition = generate_structured(client, TaskDecomposition, prompt)
        except StructuredOutputError as e:
            last_error = str(e)
            continue

        last_error = _validate(decomposition.subtasks)
        if last_error is None:
            return [
                DagNode(id=s.id, task=s.task, difficulty=s.difficulty)
                for s in decomposition.subtasks
            ]

    raise TaskDecompositionError(
        f"Could not produce a valid task decomposition for {task!r} after "
        f"{max_attempts} attempts: {last_error}"
    )
