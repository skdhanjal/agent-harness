"""Integration test for AgentHarness: proves the wiring around native
tool-calling holds together -- tool execution, approval gating,
checkpointing, tool_call_id threading, final-answer handling, context
compaction once the running message list exceeds its token budget, and
crash resume from the last checkpoint.

Uses a scripted fake ToolCallingChatClient -- no real API calls -- so the
loop is verified deterministically.
"""

import json
from pathlib import Path

import pytest

from agent_harness.framework import (
    AgentHarness,
    CheckpointNotFoundError,
    RunAlreadyFinishedError,
)
from agent_harness.gates.approval import ApprovalGate, RiskTier
from agent_harness.persistence.checkpoint import load_checkpoint
from agent_harness.persistence.retry import FatalError
from agent_harness.schemas.tool_calling import ChatTurn, ToolCallRequest
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
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


def test_harness_runs_tool_call_then_final_answer(tmp_path: Path) -> None:
    note_path = tmp_path / "notes" / "topic.md"

    client = ScriptedToolCallingClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="write_file",
                        arguments={"path": note_path.as_posix(), "content": "hello"},
                    )
                ],
            ),
            ChatTurn(content="Saved the note.", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)

    ledger = TraceLedger(tmp_path / "traces.jsonl")
    gate = ApprovalGate(approver=lambda action: True)  # auto-approve, no blocking input()

    harness = AgentHarness(
        client=client,
        tools=tools,
        gate=gate,
        ledger=ledger,
        run_id="test-run",
        instructions="Write a note and save it.",
        risk_by_tool={"write_file": RiskTier.HIGH},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    result = harness.run("write a note about testing")

    assert result == "Saved the note."
    assert note_path.read_text() == "hello"
    assert len(client.calls) == 2

    # the second call must carry the assistant tool-call turn plus a
    # tool-role result threaded back by the same tool_call_id
    second_call_messages = client.calls[1]
    assistant_msg = second_call_messages[-2]
    tool_msg = second_call_messages[-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"  # type: ignore[index]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": f"wrote 5 chars to {note_path.as_posix()}",
    }

    checkpoint = load_checkpoint("test-run", directory=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert checkpoint["phase"] == "DONE"


def test_harness_respects_denied_approval(tmp_path: Path) -> None:
    denied_path = tmp_path / "x.md"

    client = ScriptedToolCallingClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="write_file",
                        arguments={"path": denied_path.as_posix(), "content": "x"},
                    )
                ],
            ),
            ChatTurn(content="Could not save the note.", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)

    gate = ApprovalGate(approver=lambda action: False)  # deny everything

    harness = AgentHarness(
        client=client,
        tools=tools,
        gate=gate,
        ledger=TraceLedger(tmp_path / "traces.jsonl"),
        run_id="test-run-denied",
        instructions="Write a note and save it.",
        risk_by_tool={"write_file": RiskTier.HIGH},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    result = harness.run("write a note")

    assert result == "Could not save the note."
    assert not denied_path.exists()  # denied, file never written

    tool_msg = client.calls[1][-1]
    assert tool_msg["tool_call_id"] == "call_1"
    assert "DENIED" in str(tool_msg["content"])


def test_harness_compacts_context_once_over_budget(tmp_path: Path) -> None:
    """3 tool-call turns with a token budget of 1 (always exceeded) forces
    compaction as soon as there's more than `keep_recent_turns` (2) turns
    to work with -- i.e. right after the 3rd turn is appended.
    """

    def write_call(call_id: str, n: int) -> ToolCallRequest:
        return ToolCallRequest(
            id=call_id,
            name="write_file",
            arguments={"path": (tmp_path / f"note{n}.md").as_posix(), "content": f"note {n}"},
        )

    client = ScriptedToolCallingClient(
        [
            ChatTurn(content=None, tool_calls=[write_call("call_1", 1)]),
            ChatTurn(content=None, tool_calls=[write_call("call_2", 2)]),
            ChatTurn(content=None, tool_calls=[write_call("call_3", 3)]),
            ChatTurn(content="condensed summary of the first write", tool_calls=[]),
            ChatTurn(content="All notes saved.", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)
    traces_path = tmp_path / "traces.jsonl"

    harness = AgentHarness(
        client=client,
        tools=tools,
        gate=ApprovalGate(approver=lambda action: True),
        ledger=TraceLedger(traces_path),
        run_id="test-run-compaction",
        instructions="Write three notes.",
        risk_by_tool={"write_file": RiskTier.HIGH},
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_iterations=10,
        context_token_budget=1,  # any non-empty history exceeds this
    )

    result = harness.run("write three notes")

    assert result == "All notes saved."
    # 3 action calls + 1 summarize call + 1 final action call
    assert len(client.calls) == 5

    # the summarize call received exactly the stale (oldest) turn: call_1's
    # assistant message and its tool result, nothing from turns 2 or 3
    summarize_messages = client.calls[3]
    assert summarize_messages[0]["role"] == "system"
    stale_turn = json.loads(str(summarize_messages[1]["content"]))
    assert [m["role"] for m in stale_turn] == ["assistant", "tool"]
    assert stale_turn[0]["tool_calls"][0]["id"] == "call_1"
    assert stale_turn[1]["tool_call_id"] == "call_1"

    # the final action call's history has the summary in place of turn 1,
    # and turns 2-3 kept verbatim with their tool_call_ids intact
    final_messages = client.calls[4]
    assert "[Summary of 1 earlier turns]" in str(final_messages[2]["content"])
    remaining_tool_call_ids = [m["tool_call_id"] for m in final_messages if m.get("role") == "tool"]
    assert remaining_tool_call_ids == ["call_2", "call_3"]

    # compaction is observable in the trace ledger, not silent
    trace_events = [json.loads(line) for line in traces_path.read_text().splitlines()]
    compaction_events = [e for e in trace_events if e["event_type"] == "context_compaction"]
    assert len(compaction_events) == 1
    payload = compaction_events[0]["payload"]
    assert payload["tokens_after"] < payload["tokens_before"]


def test_harness_resumes_from_checkpoint_after_crash(tmp_path: Path) -> None:
    """A FatalError mid-run (not retried by with_backoff) simulates a crash.
    A fresh AgentHarness -- same run_id, same checkpoint_dir -- then resumes
    without re-executing the tool call that already ran before the crash.
    """
    note_path = tmp_path / "note.md"
    checkpoint_dir = tmp_path / "checkpoints"
    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)

    def make_harness(client: ScriptedToolCallingClient) -> AgentHarness:
        return AgentHarness(
            client=client,
            tools=tools,
            gate=ApprovalGate(approver=lambda action: True),
            ledger=TraceLedger(tmp_path / "traces.jsonl"),
            run_id="resume-run",
            instructions="Write a note and save it.",
            risk_by_tool={"write_file": RiskTier.HIGH},
            checkpoint_dir=str(checkpoint_dir),
        )

    crashing_client = ScriptedToolCallingClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="write_file",
                        arguments={"path": note_path.as_posix(), "content": "hello"},
                    )
                ],
            ),
            FatalError("simulated crash"),
        ]
    )
    with pytest.raises(FatalError):
        make_harness(crashing_client).run("write a note")

    resumed_client = ScriptedToolCallingClient([ChatTurn(content="Saved the note.", tool_calls=[])])
    result = make_harness(resumed_client).resume()

    assert result == "Saved the note."
    assert note_path.read_text() == "hello"
    assert len(resumed_client.calls) == 1  # only the post-crash call was needed

    resumed_messages = resumed_client.calls[0]
    assert resumed_messages[-2]["role"] == "assistant"
    assert resumed_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": f"wrote 5 chars to {note_path.as_posix()}",
    }


def test_resume_raises_when_no_checkpoint_exists(tmp_path: Path) -> None:
    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)
    harness = AgentHarness(
        client=ScriptedToolCallingClient([]),
        tools=tools,
        gate=ApprovalGate(approver=lambda action: True),
        ledger=TraceLedger(tmp_path / "traces.jsonl"),
        run_id="never-ran",
        instructions="x",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    with pytest.raises(CheckpointNotFoundError):
        harness.resume()


def test_resume_raises_when_run_already_finished(tmp_path: Path) -> None:
    tools = ToolRegistry()
    register_filesystem_tools(tools, root=tmp_path)
    checkpoint_dir = str(tmp_path / "checkpoints")
    client = ScriptedToolCallingClient([ChatTurn(content="All done.", tool_calls=[])])
    harness = AgentHarness(
        client=client,
        tools=tools,
        gate=ApprovalGate(approver=lambda action: True),
        ledger=TraceLedger(tmp_path / "traces.jsonl"),
        run_id="finished-run",
        instructions="x",
        checkpoint_dir=checkpoint_dir,
    )
    harness.run("a task")

    with pytest.raises(RunAlreadyFinishedError):
        harness.resume()
