"""Instrumented reproduction: logs full tool-call arguments/results and raw
model turns to a JSONL file, so a repeated/duplicate tool-call pattern can
be diagnosed instead of just counted.

Usage:
    uv run python debug_multi_tool_run.py ["task"]
"""

from __future__ import annotations

import json
import sys
import uuid

from agent_harness.framework import AgentHarness
from agent_harness.gates.approval import ApprovalGate, RiskTier
from agent_harness.schemas.openai_client import OpenAIChatClient
from agent_harness.tools.filesystem import register_filesystem_tools
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceLedger

DEFAULT_TASK = (
    "Find out how many test functions (def test_...) are defined across all "
    "test files in the tests/ directory. Write a short summary report to "
    "notes/test_summary.md listing the count per file and the grand total."
)


def main() -> None:
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    run_id = f"debug-{uuid.uuid4().hex[:8]}"
    debug_path = f"./run_output/{run_id}/debug_detail.jsonl"

    tools = ToolRegistry()
    register_filesystem_tools(tools, root=".")

    debug_log: list[dict[str, object]] = []

    real_create_action_turn = OpenAIChatClient.create_action_turn

    def logged_create_action_turn(self, messages, tools_schema):  # type: ignore[no-untyped-def]
        turn = real_create_action_turn(self, messages, tools_schema)
        debug_log.append(
            {
                "kind": "llm_turn",
                "num_input_messages": len(messages),
                "content": turn.content,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                ],
            }
        )
        return turn

    OpenAIChatClient.create_action_turn = logged_create_action_turn  # type: ignore[method-assign]

    real_execute = ToolRegistry.execute

    def logged_execute(self, name, raw_args, timeout_s=10.0):  # type: ignore[no-untyped-def]
        result = real_execute(self, name, raw_args, timeout_s)
        debug_log.append(
            {
                "kind": "tool_execute",
                "tool": name,
                "args": raw_args,
                "ok": result.ok,
                "result": result.result if result.ok else None,
                "error": result.error,
            }
        )
        return result

    ToolRegistry.execute = logged_execute  # type: ignore[method-assign]

    harness = AgentHarness(
        client=OpenAIChatClient(model="gpt-4o-mini"),
        tools=tools,
        gate=ApprovalGate(
            require_approval_at=RiskTier.HIGH,
            approver=lambda a: (print(f"[gate] approving {a.tool_name}"), True)[1],
        ),
        ledger=TraceLedger(f"./run_output/{run_id}/traces.jsonl"),
        run_id=run_id,
        instructions=(
            "You are a filesystem assistant working inside a sandboxed workspace "
            "rooted at the current project directory. You have these tools: "
            "read_file, write_file, list_files, glob_files, grep_files, file_stat, "
            "make_directory, move_file, delete_file -- all paths are relative to "
            "the workspace root. Use whichever tools the task requires, then give "
            "a final_answer describing what you did or found."
        ),
        risk_by_tool={
            "read_file": RiskTier.LOW,
            "list_files": RiskTier.LOW,
            "glob_files": RiskTier.LOW,
            "grep_files": RiskTier.LOW,
            "file_stat": RiskTier.LOW,
            "make_directory": RiskTier.LOW,
            "write_file": RiskTier.HIGH,
            "move_file": RiskTier.HIGH,
            "delete_file": RiskTier.HIGH,
        },
        max_iterations=10,
        checkpoint_dir=f"./run_output/{run_id}/checkpoints",
    )

    print(f"[run] run_id={run_id} task={task!r}")
    try:
        result = harness.run(task)
        print("\n=== FINAL ANSWER ===")
        print(result)
    except Exception as e:
        print(f"\n=== RAISED: {e!r} ===")
    finally:
        import pathlib

        pathlib.Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w") as f:
            for entry in debug_log:
                f.write(json.dumps(entry) + "\n")
        print(f"\n[debug] full detail written to {debug_path}")


if __name__ == "__main__":
    main()
