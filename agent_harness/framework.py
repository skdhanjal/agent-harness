"""The Agent Harness: composes Modules 2-9 into one runnable agent loop,
driven by native tool-calling (see agent_harness/schemas/tool_calling.py).

Nothing here is new logic -- it's wiring. The `messages` list, threaded by
`tool_call_id`, is the working memory; there's no separate memory log.
"""

from __future__ import annotations

import json
from functools import partial

from agent_harness.context.compaction import compact_messages
from agent_harness.context.tokens import count_messages_tokens, count_tokens
from agent_harness.control_flow.state_machine import AgentFSM, Phase
from agent_harness.gates.approval import ApprovalGate, PendingAction, RiskTier
from agent_harness.persistence.checkpoint import save_checkpoint
from agent_harness.persistence.retry import with_backoff
from agent_harness.schemas.tool_calling import ChatTurn, ToolCallingChatClient, ToolCallRequest
from agent_harness.tools.registry import ToolRegistry
from agent_harness.tracing.ledger import TraceEvent, TraceLedger

_SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize this earlier portion of an agent's tool-use history. Preserve "
    "concrete facts, decisions, and outcomes the agent will still need; drop "
    "raw tool-call plumbing. Be concise."
)


class MaxIterationsExceededError(RuntimeError):
    """Raised when the agent never produces a final answer within budget."""


class AgentHarness:
    def __init__(
        self,
        client: ToolCallingChatClient,
        tools: ToolRegistry,
        gate: ApprovalGate,
        ledger: TraceLedger,
        run_id: str,
        instructions: str,
        risk_by_tool: dict[str, RiskTier] | None = None,
        max_iterations: int = 5,
        checkpoint_dir: str = "./checkpoints",
        context_token_budget: int = 8_000,
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
        self.context_token_budget = context_token_budget

        self.fsm = AgentFSM(max_steps=max_iterations * 2)

    def run(self, task: str) -> str:
        self.fsm.step(Phase.EXECUTE)  # PLAN -> EXECUTE: the only legal first move
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": task},
        ]
        tools_schema = self.tools.schema_for_llm()

        for _ in range(self.max_iterations):
            turn = with_backoff(partial(self.client.create_action_turn, messages, tools_schema))
            self.ledger.log(
                TraceEvent(
                    run_id=self.run_id,
                    event_type="llm_call",
                    payload={"content": turn.content, "num_tool_calls": len(turn.tool_calls)},
                )
            )
            self._checkpoint()

            if not turn.tool_calls:
                self.fsm.step(Phase.REVIEW)
                self.fsm.step(Phase.DONE)
                self._checkpoint()
                return turn.content or ""

            messages.append(self._assistant_message(turn))
            for call in turn.tool_calls:
                messages.append(self._execute_tool_call(call))

            messages = self._compact_if_needed(messages)

            self.fsm.step(Phase.EXECUTE)
            self._checkpoint()

        raise MaxIterationsExceededError(f"No final answer within {self.max_iterations} iterations")

    @staticmethod
    def _assistant_message(turn: ChatTurn) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": turn.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in turn.tool_calls
            ],
        }

    def _execute_tool_call(self, call: ToolCallRequest) -> dict[str, object]:
        risk = self.risk_by_tool.get(call.name, RiskTier.LOW)
        pending = PendingAction(
            tool_name=call.name, args=call.arguments, risk=risk, run_id=self.run_id
        )

        if not self.gate.check(pending):
            self.ledger.log(
                TraceEvent(
                    run_id=self.run_id,
                    event_type="gate_decision",
                    payload={"tool": call.name, "approved": False},
                )
            )
            denial = f"DENIED: approval required for '{call.name}' (risk={risk.name}) not granted"
            return {"role": "tool", "tool_call_id": call.id, "content": denial}

        result = self.tools.execute(call.name, call.arguments)
        self.ledger.log(
            TraceEvent(
                run_id=self.run_id,
                event_type="tool_call",
                payload={"tool": call.name, "ok": result.ok},
            )
        )
        outcome = result.result if result.ok else f"ERROR: {result.error}"
        return {"role": "tool", "tool_call_id": call.id, "content": str(outcome)}

    def _compact_if_needed(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        tokens_before = count_messages_tokens(messages, count_tokens)
        if tokens_before <= self.context_token_budget:
            return messages

        compacted = compact_messages(
            messages,
            token_budget=self.context_token_budget,
            count_tokens=count_tokens,
            summarize=self._summarize_turns,
        )
        if compacted is not messages:
            self.ledger.log(
                TraceEvent(
                    run_id=self.run_id,
                    event_type="context_compaction",
                    payload={
                        "tokens_before": tokens_before,
                        "tokens_after": count_messages_tokens(compacted, count_tokens),
                    },
                )
            )
        return compacted

    def _summarize_turns(self, turn_messages: list[dict[str, object]]) -> str:
        prompt: list[dict[str, object]] = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(turn_messages)},
        ]
        turn = self.client.create_action_turn(prompt, [])
        return turn.content or ""

    def _checkpoint(self) -> None:
        save_checkpoint(
            self.run_id,
            {"phase": self.fsm.phase.name, "step_count": self.fsm.step_count},
            directory=self.checkpoint_dir,
        )
