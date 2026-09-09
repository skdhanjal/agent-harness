"""The Agent Harness: composes Modules 1-9 into one runnable agent loop.

Nothing here is new logic -- it's wiring. The model proposes one
AgentAction per turn; each module gets a chance to validate, gate, log,
or persist it before the loop continues. The model never picks a tool
unchecked and never declares itself done unchecked.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from agent_harness.context.builder import ContextBuilder
from agent_harness.context.tokens import count_tokens as real_count_tokens
from agent_harness.control_flow.state_machine import AgentFSM, Phase
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.memory.sliding_window import SlidingWindowMemory
from agent_harness.persistence.checkpoint import save_checkpoint
from agent_harness.persistence.retry import with_backoff
from agent_harness.schemas.actions import ActionType, AgentAction
from agent_harness.schemas.structured import ChatClient, generate_structured
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceEvent, TraceLedger


class MaxIterationsExceededError(RuntimeError):
    """Raised when the agent never produces a final answer within budget."""


class AgentHarness:
    def __init__(
        self,
        client: ChatClient,
        tools: ToolRegistry,
        gate: ApprovalGate,
        ledger: TraceLedger,
        run_id: str,
        instructions: str,
        risk_by_tool: dict[str, RiskTier] | None = None,
        max_iterations: int = 5,
        checkpoint_dir: str = "./checkpoints",
        count_tokens_fn: Callable[[str], int] = real_count_tokens,
    ) -> None:
        self.client = client
        self.tools = tools
        self.gate = gate
        self.ledger = ledger
        self.run_id = run_id
        self.instructions = instructions
        self.risk_by_tool = risk_by_tool or {}
        self.max_iterations = max_iterations
        self.checkpoint_dir = checkpoint_dir
        self._count_tokens = count_tokens_fn

        self.fsm = AgentFSM(max_steps=max_iterations * 2)
        self.memory = SlidingWindowMemory(max_turns=20)

    def run(self, task: str) -> str:
        self.fsm.step(Phase.EXECUTE)  # PLAN -> EXECUTE: the only legal first move
        self.memory.add(f"TASK: {task}")

        for _ in range(self.max_iterations):
            context = self._build_context(task)
            tools_schema = self.tools.schema_for_llm()

            action = with_backoff(
                partial(
                    generate_structured,
                    self.client,
                    AgentAction,
                    context,
                    max_repairs=2,
                    tools=tools_schema,
                )
            )
            self.ledger.log(
                TraceEvent(
                    run_id=self.run_id,
                    event_type="llm_call",
                    payload={"reasoning": action.reasoning, "action": action.action_type.value},
                )
            )
            self._checkpoint()

            if action.action_type == ActionType.FINAL_ANSWER:
                self.fsm.step(Phase.REVIEW)
                self.fsm.step(Phase.DONE)
                self._checkpoint()
                return action.final_answer or ""

            self._handle_tool_call(action)
            self.fsm.step(Phase.EXECUTE)
            self._checkpoint()

        raise MaxIterationsExceededError(f"No final answer within {self.max_iterations} iterations")

    def _handle_tool_call(self, action: AgentAction) -> None:
        raw_tool_name = action.tool_name or ""
        tool_name = raw_tool_name.split(".")[-1]
        risk = self.risk_by_tool.get(tool_name, RiskTier.LOW)
        pending = PendingAction(
            tool_name=tool_name, args=action.tool_args, risk=risk, run_id=self.run_id
        )

        print("Tool call pending action", pending)

        if not self.gate.check(pending):
            print("[Tool approval required]", pending.tool_name, pending.risk)
            self.memory.add(f"DENIED: {tool_name} (risk={risk.name})")
            self.ledger.log(
                TraceEvent(
                    run_id=self.run_id,
                    event_type="gate_decision",
                    payload={"tool": tool_name, "approved": False},
                )
            )
            return

        result = self.tools.execute(tool_name, action.tool_args)
        self.ledger.log(
            TraceEvent(
                run_id=self.run_id,
                event_type="tool_call",
                payload={"tool": tool_name, "ok": result.ok},
            )
        )
        outcome = result.result if result.ok else result.error
        self.memory.add(f"TOOL RESULT ({tool_name}): {outcome}")

    def _build_context(self, task: str) -> str:
        builder = ContextBuilder(token_budget=2000, count_tokens=self._count_tokens)
        builder.add("instructions", self.instructions, priority=1)
        builder.add("task", task, priority=2)
        builder.add("history", "\n".join(self.memory.get_all()), priority=3)
        return builder.build()

    def _checkpoint(self) -> None:
        save_checkpoint(
            self.run_id,
            {"phase": self.fsm.phase.name, "step_count": self.fsm.step_count},
            directory=self.checkpoint_dir,
        )
