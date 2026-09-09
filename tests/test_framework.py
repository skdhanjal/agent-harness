"""Integration test for AgentHarness: proves Modules 1-9 wire together.

Uses a scripted fake ChatClient and a fake token counter -- no real API
calls, no tiktoken network dependency -- so the wiring logic (tool
execution, approval gating, checkpointing, final-answer handling) is
verified deterministically.
"""

from pathlib import Path

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, RiskTier
from agent_harness.persistence.checkpoint import load_checkpoint
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger


def _word_count(text: str) -> int:
    return len(text.split())


class ScriptedChatClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def test_harness_runs_tool_call_then_final_answer(tmp_path: Path) -> None:
    note_path = tmp_path / "notes" / "topic.md"

    tool_call_response = (
        '{"reasoning": "I will save the note first", '
        '"action_type": "tool_call", "tool_name": "write_file", '
        f'"tool_args": {{"path": "{note_path.as_posix()}", "content": "hello"}}, '
        '"final_answer": null}'
    )
    final_response = (
        '{"reasoning": "Note saved, done", "action_type": "final_answer", '
        '"tool_name": null, "tool_args": {}, "final_answer": "Saved the note."}'
    )
    client = ScriptedChatClient([tool_call_response, final_response])

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
        count_tokens_fn=_word_count,
    )

    result = harness.run("write a note about testing")

    assert result == "Saved the note."
    assert note_path.read_text() == "hello"
    assert client.call_count == 2

    checkpoint = load_checkpoint("test-run", directory=tmp_path / "checkpoints")
    assert checkpoint is not None
    assert checkpoint["phase"] == "DONE"


def test_harness_respects_denied_approval(tmp_path: Path) -> None:
    denied_path = tmp_path / "x.md"

    tool_call_response = (
        '{"reasoning": "Attempting write", "action_type": "tool_call", '
        '"tool_name": "write_file", '
        f'"tool_args": {{"path": "{denied_path.as_posix()}", "content": "x"}}, '
        '"final_answer": null}'
    )
    final_response = (
        '{"reasoning": "Denied, giving up", "action_type": "final_answer", '
        '"tool_name": null, "tool_args": {}, "final_answer": "Could not save the note."}'
    )
    client = ScriptedChatClient([tool_call_response, final_response])

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
        count_tokens_fn=_word_count,
    )

    result = harness.run("write a note")

    assert result == "Could not save the note."
    assert not denied_path.exists()  # denied, file never written
