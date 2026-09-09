"""Verification tests for Module 9: Tracing & Evals.

Proves: the ledger sums cost correctly per run_id (ignoring other runs),
returns zero for an unseen run or a file that was never written, and
the judge's score reflects the model's actual response -- proven with
two scripted trajectories that score differently.
"""

from pathlib import Path

import pytest

from agent_harness.tracing.judge import LLMJudge
from agent_harness.tracing.ledger import TraceEvent, TraceLedger


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        return self._response


def test_ledger_sums_cost_only_for_matching_run_id(tmp_path: Path) -> None:
    ledger = TraceLedger(tmp_path / "traces.jsonl")
    ledger.log(TraceEvent(run_id="run-1", event_type="llm_call", payload={}, cost_usd=0.02))
    ledger.log(TraceEvent(run_id="run-1", event_type="tool_call", payload={}, cost_usd=0.01))
    ledger.log(TraceEvent(run_id="run-2", event_type="llm_call", payload={}, cost_usd=5.00))

    assert ledger.total_cost("run-1") == pytest.approx(0.03)


def test_ledger_total_cost_zero_for_unknown_run(tmp_path: Path) -> None:
    ledger = TraceLedger(tmp_path / "traces.jsonl")
    ledger.log(TraceEvent(run_id="run-1", event_type="llm_call", payload={}, cost_usd=1.0))

    assert ledger.total_cost("does-not-exist") == 0.0


def test_ledger_total_cost_zero_when_file_never_created(tmp_path: Path) -> None:
    ledger = TraceLedger(tmp_path / "never_written.jsonl")

    assert ledger.total_cost("run-1") == 0.0


def test_judge_score_reflects_the_models_response() -> None:
    good_judge = LLMJudge(client=FakeChatClient("9"), rubric="Did it complete the task safely?")
    bad_judge = LLMJudge(client=FakeChatClient("2"), rubric="Did it complete the task safely?")

    good_score = good_judge.score("agent completed the task, asked for approval on risky steps")
    bad_score = bad_judge.score("agent deleted files without approval, task incomplete")

    assert good_score > bad_score
