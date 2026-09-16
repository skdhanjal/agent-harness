"""Wire up AgentHarness end-to-end and run it against a real task.

Approval is auto-granted for a non-interactive run; swap `approver` for a
blocking CLI prompt for human-in-the-loop approval.

Usage:
    uv run python run_agent.py ["task"]
"""

from __future__ import annotations

import sys
import uuid

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.memory.durable_store import DurableStateStore
from agent_harness.memory.tools import register_memory_tools
from agent_harness.prompts.template import PromptRegistry, PromptTemplate
from agent_harness.schemas.openai_client import OpenAIChatClient
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger

# Shared across runs (unlike run_output/{run_id}/...) -- this is the whole point
# of durable memory: facts survive past the process that wrote them.
MEMORY_PATH = "./agent_memory/facts.json"

DEFAULT_TASK = (
    "Write study notes about how photosynthesis works and save them as "
    "Markdown to notes/photosynthesis.md."
)

# reads/list/stat/mkdir are safe; write/move/delete are gated
RISK_BY_TOOL = {
    "read_file": RiskTier.LOW,
    "list_files": RiskTier.LOW,
    "glob_files": RiskTier.LOW,
    "grep_files": RiskTier.LOW,
    "file_stat": RiskTier.LOW,
    "make_directory": RiskTier.LOW,
    "write_file": RiskTier.HIGH,
    "move_file": RiskTier.HIGH,
    "delete_file": RiskTier.HIGH,
    # memory is a self-contained store, not the user's own files -- writes
    # here get a lighter tier than a filesystem write
    "recall_fact": RiskTier.LOW,
    "list_facts": RiskTier.LOW,
    "remember_fact": RiskTier.MEDIUM,
}

PROMPTS = PromptRegistry()
PROMPTS.register(
    PromptTemplate(
        name="filesystem_agent",
        version="v1",
        system_tmpl=(
            "You are a filesystem assistant working inside a sandboxed workspace "
            "rooted at the current project directory. You have these tools: "
            "read_file, write_file, list_files, glob_files, grep_files, file_stat, "
            "make_directory, move_file, delete_file -- all paths are relative to "
            "the workspace root. You also have remember_fact, recall_fact, and "
            "list_facts, which persist key-value facts across separate runs, not "
            "just this conversation -- use them for things worth knowing next "
            "time (user preferences, stable project facts), not for scratch "
            "state from this task alone. Use whichever tools the task requires, "
            "then give a final_answer describing what you did or found."
        ),
        user_tmpl="$task",
    )
)


def auto_approve(action: PendingAction) -> bool:
    print(f"[gate] auto-approving {action.tool_name}({action.args}) risk={action.risk.name}")
    return True


def main() -> None:
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=".")
    register_memory_tools(tools, DurableStateStore(MEMORY_PATH))

    prompt = PROMPTS.get("filesystem_agent").render(task=task)

    harness = AgentHarness(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        tools=tools,
        gate=ApprovalGate(
            require_approval_at=RiskTier.HIGH,
            approver=auto_approve,
            pending_dir=f"./run_output/{run_id}/pending_actions",
        ),
        ledger=TraceLedger(f"./run_output/{run_id}/traces.jsonl"),
        run_id=run_id,
        instructions=prompt.system,
        risk_by_tool=RISK_BY_TOOL,
        max_iterations=6,
        checkpoint_dir=f"./run_output/{run_id}/checkpoints",
        prompt_id=prompt.prompt_id,
    )

    print(f"[run] run_id={run_id} task={task!r} prompt_id={prompt.prompt_id}")
    result = harness.run(prompt.user)

    print("\n=== FINAL ANSWER ===")
    print(result)


if __name__ == "__main__":
    main()
