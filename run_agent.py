"""Wire up AgentHarness end-to-end and run it against a real task.

Uses the real OpenAI-backed ChatClient (Module 1) and the filesystem tools
(Module 2) to have the agent research a topic and save notes to disk via
`write_file`, gated behind approval (Module 8) -- auto-approved here for a
non-interactive run; swap `approver` for a blocking CLI prompt to see the
human-in-the-loop gate in action.

Usage:
    uv run python run_agent.py ["topic"]
"""

from __future__ import annotations

import re
import sys
import uuid

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.schemas.openai_client import OpenAIChatClient
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger

DEFAULT_TOPIC = "how photosynthesis works"


def auto_approve(action: PendingAction) -> bool:
    print(f"[gate] auto-approving {action.tool_name}({action.args}) risk={action.risk.name}")
    return True


def main() -> None:
    topic = " ".join(sys.argv[1:]) or DEFAULT_TOPIC
    run_id = f"notes-{uuid.uuid4().hex[:8]}"
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    notes_path = f"notes/{slug}.md"

    tools = ToolRegistry()
    register_filesystem_tools(tools)

    harness = AgentHarness(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        tools=tools,
        gate=ApprovalGate(require_approval_at=RiskTier.HIGH, approver=auto_approve),
        ledger=TraceLedger(f"./run_output/{run_id}/traces.jsonl"),
        run_id=run_id,
        instructions=(
            "You are a research assistant. Write clear, well-organized study "
            "notes in Markdown on the given topic. Save them with the "
            f"write_file tool to the path '{notes_path}'. Once saved, give a "
            "final_answer summarizing what the notes cover and confirming "
            "the file path."
        ),
        risk_by_tool={"write_file": RiskTier.HIGH, "list_files": RiskTier.LOW},
        max_iterations=6,
        checkpoint_dir=f"./run_output/{run_id}/checkpoints",
    )

    print(f"[run] run_id={run_id} topic={topic!r} -> {notes_path}")
    result = harness.run(f"Write study notes about: {topic}")

    print("\n=== FINAL ANSWER ===")
    print(result)


if __name__ == "__main__":
    main()
