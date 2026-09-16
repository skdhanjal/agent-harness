"""Wire up OrchestrationDriver end-to-end: decompose one task string into a
DAG of independent subtasks, run each as its own full AgentHarness, routed
to a model tier by difficulty and escalated one tier up if it fails.

Each node gets its own run_id, trace file, and checkpoint dir, nested under
the parent orchestration run -- see build_harness() below.

Usage:
    uv run python run_orchestrated.py ["task"]
"""

from __future__ import annotations

import sys
import uuid

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.memory.durable_store import DurableStateStore
from agent_harness.memory.tools import register_memory_tools
from agent_harness.orchestration.decomposition import decompose_task
from agent_harness.orchestration.driver import OrchestrationDriver
from agent_harness.orchestration.router import ModelRouter, ModelTier
from agent_harness.orchestration.spawner import DagNode
from agent_harness.schemas.openai_client import OpenAIChatClient
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger

# Same store run_agent.py uses -- memory is shared across every kind of run,
# single-agent or orchestrated.
MEMORY_PATH = "./agent_memory/facts.json"

# capability_rank order matters (ModelRouter sorts by it); `name` doubles as
# the literal OpenAI model id passed to OpenAIChatClient.
TIERS = [
    ModelTier(name="gpt-4o-mini", cost_per_1k=0.00015, capability_rank=0),
    ModelTier(name="gpt-4o", cost_per_1k=0.0025, capability_rank=1),
]

DEFAULT_TASK = (
    "Write study notes about photosynthesis, mitosis, and the laws of "
    "thermodynamics, saving each topic as its own Markdown file under notes/."
)

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
    "recall_fact": RiskTier.LOW,
    "list_facts": RiskTier.LOW,
    "remember_fact": RiskTier.MEDIUM,
}

INSTRUCTIONS = (
    "You are a filesystem assistant working inside a sandboxed workspace rooted "
    "at the current project directory. You have these tools: read_file, "
    "write_file, list_files, glob_files, grep_files, file_stat, make_directory, "
    "move_file, delete_file, remember_fact, recall_fact, list_facts -- all paths "
    "are relative to the workspace root. Use whichever tools the task requires, "
    "then give a final_answer describing what you did or found."
)


def auto_approve(action: PendingAction) -> bool:
    print(f"[gate] auto-approving {action.tool_name}({action.args}) risk={action.risk.name}")
    return True


def build_harness(node: DagNode, tier: ModelTier, parent_run_id: str) -> AgentHarness:
    """One full harness per node -- AgentFSM/messages aren't safe to share
    across the spawner's concurrent threads, so each node gets its own.
    """
    run_id = f"{parent_run_id}::{node.id}"
    output_dir = f"./run_output/{parent_run_id}/subagents/{node.id}"

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=".")
    register_memory_tools(tools, DurableStateStore(MEMORY_PATH))

    return AgentHarness(
        client=OpenAIChatClient(model=tier.name),
        tools=tools,
        gate=ApprovalGate(
            require_approval_at=RiskTier.HIGH,
            approver=auto_approve,
            pending_dir=f"{output_dir}/pending_actions",
        ),
        ledger=TraceLedger(f"{output_dir}/traces.jsonl"),
        run_id=run_id,
        instructions=INSTRUCTIONS,
        risk_by_tool=RISK_BY_TOOL,
        max_iterations=6,
        checkpoint_dir=f"{output_dir}/checkpoints",
    )


def main() -> None:
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    parent_run_id = f"orchestrated-{uuid.uuid4().hex[:8]}"

    # Cheap tier only, for the one-shot decomposition call -- not one of the
    # per-node harness clients, which are built per-tier in build_harness().
    nodes = decompose_task(task, OpenAIChatClient(model=TIERS[0].name))
    print(f"[run] parent_run_id={parent_run_id} task={task!r} nodes={[n.id for n in nodes]}")

    driver = OrchestrationDriver(
        router=ModelRouter(TIERS),
        harness_builder=lambda node, tier: build_harness(node, tier, parent_run_id),
        max_workers=4,
        max_attempts=3,
    )
    results = driver.run(nodes)

    print("\n=== RESULTS ===")
    for node_id, node_result in results.items():
        status = "OK" if node_result.ok else "FAILED"
        print(
            f"[{status}] {node_id} (tier={node_result.tier_used}, attempts={node_result.attempts})"
        )
        print(f"  {node_result.result if node_result.ok else node_result.error}")


if __name__ == "__main__":
    main()
