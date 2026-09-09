"""Integration test for AgentHarness: proves the wiring around native
tool-calling holds together -- tool execution, approval gating,
checkpointing, tool_call_id threading, and final-answer handling.

Uses a scripted fake ToolCallingChatClient -- no real API calls -- so the
loop is verified deterministically.
"""

from pathlib import Path

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, RiskTier
from agent_harness.persistence.checkpoint import load_checkpoint
from agent_harness.schemas.tool_calling import ChatTurn, ToolCallRequest
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger


class ScriptedToolCallingClient:
    def __init__(self, turns: list[ChatTurn]) -> None:
        self._turns = turns
        self.calls: list[list[dict[str, object]]] = []

    def create_action_turn(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ChatTurn:
        self.calls.append(list(messages))
        return self._turns[len(self.calls) - 1]


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
    register_filesystem_tools(tools)

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
    register_filesystem_tools(tools)

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
