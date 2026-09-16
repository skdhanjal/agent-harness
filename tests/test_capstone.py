"""Capstone integration test: everything test_framework.py and
test_module_09_tracing_evals.py prove individually, exercised together in
one continuous run.

Proves: a multi-step task through a real ApprovalGate and a forced
checkpoint, a simulated crash-and-resume via load_checkpoint, a structured
final answer that validates against a pydantic schema, a gapless trace
ledger with correct total cost, and an LLMJudge pass across 3 deterministic
runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.persistence.checkpoint import load_checkpoint
from agent_harness.persistence.retry import FatalError
from agent_harness.schemas.tool_calling import ChatTurn, ToolCallRequest
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.judge import LLMJudge
from agent_harness.tracing.ledger import TraceLedger


class ScriptedToolCallingClient:
    def __init__(self, turns: list[ChatTurn | Exception]) -> None:
        self._turns = turns
        self.calls: list[list[dict[str, object]]] = []

    def create_action_turn(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        self.calls.append(list(messages))
        turn = self._turns[len(self.calls) - 1]
        if isinstance(turn, Exception):
            raise turn
        return turn


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def create_completion(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]]
    ) -> str:
        return self._response


class RunSummary(BaseModel):
    status: str
    files_written: list[str]


def test_capstone_survives_crash_and_resume_with_gapless_trace_and_valid_output(
    tmp_path: Path,
) -> None:
    note_path1 = tmp_path / "note1.md"
    note_path2 = tmp_path / "note2.md"
    checkpoint_dir = tmp_path / "checkpoints"
    traces_path = tmp_path / "traces.jsonl"
    run_id = "capstone-run"
    prompt_id = "capstone_agent@v1"

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)

    def make_harness(
        client: ScriptedToolCallingClient, approver: Callable[[PendingAction], bool]
    ) -> AgentHarness:
        return AgentHarness(
            client=client,
            tools=tools,
            gate=ApprovalGate(approver=approver, pending_dir=str(tmp_path / "pending")),
            ledger=TraceLedger(traces_path),
            run_id=run_id,
            instructions="Write two notes and save them, then report what you did as JSON.",
            risk_by_tool={"write_file": RiskTier.HIGH},
            checkpoint_dir=str(checkpoint_dir),
            prompt_id=prompt_id,
        )

    class CrashOnSecondApproval:
        """Approves the first call, then crashes while approving the
        second -- stand-in for a process crash mid-batch, same technique
        as test_framework.py's mid-batch resume test.
        """

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, action: PendingAction) -> bool:
            self.calls += 1
            if self.calls == 2:
                raise FatalError("simulated crash while waiting on approval")
            return True

    crashing_client = ScriptedToolCallingClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="write_file",
                        arguments={"path": note_path1.as_posix(), "content": "note one"},
                    ),
                    ToolCallRequest(
                        id="call_2",
                        name="write_file",
                        arguments={"path": note_path2.as_posix(), "content": "note two"},
                    ),
                ],
                tokens_in=100,
                tokens_out=20,
                cost_usd=0.001,
            )
        ]
    )

    with pytest.raises(FatalError):
        make_harness(crashing_client, CrashOnSecondApproval()).run("write two notes")

    # Forced checkpoint: the first tool call's result is durably saved: the
    # second call's tool_call_id has no answer yet, and the run never
    # reached a terminal phase.
    checkpoint = load_checkpoint(run_id, directory=checkpoint_dir)
    assert checkpoint is not None
    assert checkpoint["phase"] not in ("DONE", "FAILED")
    checkpointed_messages = checkpoint["messages"]
    tool_messages = [m for m in checkpointed_messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1"]

    resumed_client = ScriptedToolCallingClient(
        [
            ChatTurn(
                content=json.dumps(
                    {"status": "done", "files_written": [note_path1.name, note_path2.name]}
                ),
                tool_calls=[],
                tokens_in=50,
                tokens_out=10,
                cost_usd=0.0005,
            )
        ]
    )
    result = make_harness(resumed_client, lambda action: True).resume()

    # Only the interrupted second call plus the final-answer turn ran --
    # not a re-ask of the model for a turn it already had the answer to.
    assert len(resumed_client.calls) == 1
    assert note_path1.read_text() == "note one"
    assert note_path2.read_text() == "note two"

    # Final output schema validation.
    summary = RunSummary.model_validate_json(result)
    assert summary.status == "done"
    assert set(summary.files_written) == {note_path1.name, note_path2.name}

    # Gapless trace ledger with correct total cost.
    trace_events = [json.loads(line) for line in traces_path.read_text().splitlines()]

    run_starts = [e for e in trace_events if e["event_type"] == "run_start"]
    assert len(run_starts) == 1
    assert run_starts[0]["payload"]["prompt_id"] == prompt_id

    llm_calls = [e for e in trace_events if e["event_type"] == "llm_call"]
    assert len(llm_calls) == 2  # the pre-crash batch turn, the post-resume final turn

    tool_call_events = [e for e in trace_events if e["event_type"] == "tool_call"]
    assert len(tool_call_events) == 2  # both writes, one pre-crash and one post-resume
    assert all(e["payload"]["ok"] for e in tool_call_events)

    assert TraceLedger(traces_path).total_cost(run_id) == 0.001 + 0.0005


def test_capstone_llm_judge_scores_three_deterministic_runs_consistently(tmp_path: Path) -> None:
    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)

    results: list[str] = []
    trajectories: list[str] = []

    for i in range(3):
        client = ScriptedToolCallingClient([ChatTurn(content="All done.", tool_calls=[])])
        harness = AgentHarness(
            client=client,
            tools=tools,
            gate=ApprovalGate(
                approver=lambda action: True, pending_dir=str(tmp_path / f"pending-{i}")
            ),
            ledger=TraceLedger(tmp_path / f"traces-{i}.jsonl"),
            run_id=f"capstone-det-{i}",
            instructions="Say hello.",
            checkpoint_dir=str(tmp_path / f"checkpoints-{i}"),
        )
        result = harness.run("say hello")
        results.append(result)
        # Same trajectory shape AgentHarness._summarize_turns already
        # builds (json.dumps of the run's messages).
        trajectories.append(
            json.dumps([*client.calls[0], {"role": "assistant", "content": result}])
        )

    assert len(set(results)) == 1  # same scripted client -> deterministic output every run

    judge = LLMJudge(client=FakeChatClient("8"), rubric="Did the agent complete the task?")
    scores = [judge.score(trajectory) for trajectory in trajectories]

    assert scores == [8, 8, 8]
