# Production Readiness TODO

`docs/agent_harness_roadmap.md` is the spec each module was built against, and
every module passes its own verification test. This document is different:
it's the gap between "10 modules pass their own tests in isolation" and "this
is safe to run unattended against real work." Each item below was confirmed
against the actual code (file/line references included), not assumed from
the module's intent.

Ordered by priority. Items are independent — pick any one up without needing
the others done first, unless a "depends on" note says otherwise.

## Fixed

### Unbounded context growth — fixed
`compact_messages` (`agent_harness/context/compaction.py`) is wired into
`AgentHarness._compact_if_needed`, checked once per loop iteration. Full
design rationale — why turn-pairs are the compaction unit, why summarize
instead of drop, why `self.client` is reused instead of a second injected
dependency, why the summary lands as `role: user`, why the budget defaults
to 8,000 tokens, and the one known residual limitation (a single
over-budget turn can't be shrunk by this pass) — is written up in
`docs/agent_harness_roadmap.md`'s Module 4 section, not duplicated here.
Tested in `tests/test_module_04_context_construction.py` and
`tests/test_framework.py::test_harness_compacts_context_once_over_budget`.

### Checkpoint resume doesn't work — fixed
`AgentHarness.resume()` (`agent_harness/framework.py`) now loads the last
checkpoint for `self.run_id`, restores FSM state, and re-enters the same
loop `run()` uses. `_checkpoint()` now saves `messages` too, not just
`{phase, step_count}`. Full design rationale — why `messages` lives inline
in the checkpoint file, why `resume()` always reads from disk instead of
trusting in-memory state, why there's no migration path for old checkpoint
files, and how the FSM's `step_count` remains the real cross-process
iteration ceiling — is written up in `docs/agent_harness_roadmap.md`'s
Module 7 section. Tested in `tests/test_framework.py`
(`test_harness_resumes_from_checkpoint_after_crash` and the two
`test_resume_raises_*` cases).

### Cost/token tracking is always zero — fixed
`ChatTurn` (`agent_harness/schemas/tool_calling.py`) now carries
`tokens_in`/`tokens_out`/`cost_usd`, populated by
`OpenAIChatClient.create_action_turn` from the real `response.usage` and a
small per-model price table, and `AgentHarness._run_loop` threads all three
onto every `llm_call` `TraceEvent` it logs. `TraceLedger.total_cost()` now
sums real numbers for a real run instead of always `0.0`. Full design
rationale — why the pricing table lives in `openai_client.py` instead of a
shared `tracing` module, why `create_completion` (the older Module 1
protocol) was deliberately left out of this fix, and the fallback behavior
for an unpriced model or a missing `usage` field — is written up in
`docs/agent_harness_roadmap.md`'s Module 9 section, not duplicated here.
Tested in `tests/test_openai_client.py` (usage capture and cost math against
a mocked SDK response) and `tests/test_framework.py`
(`test_harness_records_token_usage_and_cost_from_chat_turn`, proving the
harness threads a turn's numbers through to the ledger unchanged).

### Retry policy too permissive — fixed
`OpenAIChatClient` (`agent_harness/schemas/openai_client.py`) now catches the
OpenAI SDK's non-retryable client errors (`AuthenticationError`,
`BadRequestError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError`,
`UnprocessableEntityError`) around both `create_completion` and
`create_action_turn`'s API calls, and re-raises them as `FatalError` so
`with_backoff` fails fast instead of retrying 5x. `RateLimitError`,
`InternalServerError`, and connection/timeout errors are left alone — those
stay retryable, since that's exactly what backoff is for. Tested in
`tests/test_openai_client.py`
(`test_create_action_turn_reraises_auth_error_as_fatal` and
`test_create_action_turn_still_retries_rate_limit_error`).

## High (core production gaps)

### 1. Approval gate has no crash durability
`ApprovalGate` (`agent_harness/gates/approval.py`) is purely synchronous —
`check()` calls the approver inline; the default blocks on `input()`. No
`PendingAction` is ever persisted, and there's no linkage to checkpointing.
If the process crashes while waiting on approval, the pending decision is
gone. The roadmap's own Module 8 spec calls for this explicitly ("implemented
first as a blocking CLI prompt, later as an async webhook/queue") but the
"later" part was never built. Checkpoint resume (now fixed, see above) means
there's somewhere to restore a "waiting on approval" state *to* — but nothing
persists that state as a `PendingAction` in the first place yet.

## Medium (scale/quality)

### 2. Memory tiers built but unused
`DurableStateStore` and `VectorMemory` (`agent_harness/memory/`) are fully
implemented and independently tested but never imported by `framework.py` or
`run_agent.py`. `AgentHarness` has no memory-store fields; every run starts
from a blank slate beyond its own `messages` list, and nothing survives
across separate runs (user preferences, facts learned in a previous session,
semantic recall of prior work).

Fix shape: needs a product decision first — what should actually be
persisted across runs for *this* harness's use case — before it's worth
wiring in mechanically.

### 3. Orchestration not composed into a real driver
`ModelRouter`/`SubAgentSpawner` (`agent_harness/orchestration/`) are only
exercised by `tests/test_module_10_orchestration.py`. `AgentHarness.run()`
has signature `run(self, task: str)` — no `run_id` override — so the
roadmap's own spawner sketch (`agent.run(node.task, run_id=f"{parent}::{node.id}")`,
`docs/agent_harness_roadmap.md`) can't work as written: every spawned
sub-agent would log under the same `run_id` fixed at harness construction,
not a distinct per-node id. There's no driver script anywhere that
decomposes a task into a DAG and actually routes/spawns.

Fix shape: add an optional `run_id` param to `run()` (falling back to
`self.run_id`), then build a real orchestrator module that uses `ModelRouter`
to pick a tier per node and `SubAgentSpawner` to run them, each with its own
trace file.

### 4. Prompts aren't actually versioned
`PromptTemplate`/`PromptRegistry` (`agent_harness/prompts/template.py`) exist
and are tested, but the real system/task instructions are raw f-strings
built inline in `run_agent.py` (see `RISK_BY_TOOL`'s neighboring
`instructions=(...)` block) — not routed through `PromptRegistry.render()`.
No `TraceEvent` ever records which prompt version produced a given run, so
Module 3's explicit goal (know which prompt version produced which output)
isn't actually achieved outside the module's own tests.

## Lower priority (hardening/DX)

### 5. Missing capstone integration test
`docs/agent_harness_roadmap.md`'s closing step describes
`tests/test_capstone.py`: a multi-step task through a real approval gate and
forced checkpoint, a simulated crash-and-resume via `load_checkpoint`, final
output schema validation, a gapless trace ledger with correct total cost, and
an `LLMJudge` pass across 3 deterministic runs. This file still doesn't
exist. Checkpoint resume and cost tracking (both above) are done, so this
test is now fully writable against real behavior — nothing else on this list
blocks it anymore.

### 6. No CI coverage-threshold enforcement
`pyproject.toml`'s `[tool.pytest.ini_options].addopts` reports coverage
(`--cov=agent_harness --cov-report=term-missing`) but has no
`--cov-fail-under=N`, so `.github/workflows/ci.yml` can't fail a PR that
regresses coverage — it only fails on outright test failures.

### 7. No installable CLI / packaging
No `[project.scripts]` table in `pyproject.toml`. `run_agent.py` (invoked via
`uv run python run_agent.py "task"`) is the only entry point — no
`agent-harness run "task"` command, no config file for model/risk-tier/root
selection instead of editing the script directly.

### 8. Single-provider lock-in
`ChatClient` (`agent_harness/schemas/structured.py`) and
`ToolCallingChatClient` (`agent_harness/schemas/tool_calling.py`) are both
provider-agnostic `Protocol`s by design, but `OpenAIChatClient` is the only
implementation that exists. Nothing proves the abstraction actually holds
for a second provider, and there's no runtime provider-selection mechanism.

### 9. `TraceLedger` has no rotation/retention policy
A failure mode the roadmap itself calls out under Module 9 ("trace volume
growing unbounded... becomes an ops problem") but never addresses — `TraceLedger.log()`
just appends to one JSONL file forever.

### 10. No process-wide concurrency cap across parallel sub-agent runs
`SubAgentSpawner`'s `max_workers` (`agent_harness/orchestration/spawner.py`)
caps concurrency *within one spawner instance*, but nothing caps total
concurrent API spend if multiple spawners/harnesses run in the same process
or host — the "fork bomb of agents calling agents" failure mode the roadmap
names for Module 10. Only relevant once #3 has a real multi-agent driver to
run more than one spawner at a time.
