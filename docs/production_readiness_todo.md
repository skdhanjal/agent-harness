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

## Critical (correctness/safety)

### 1. Checkpoint resume doesn't actually work
`load_checkpoint` (`agent_harness/persistence/checkpoint.py`) is never called
by any non-test code — only by `tests/test_framework.py` and
`tests/test_module_07_checkpointing.py`. `_checkpoint()`
(`agent_harness/framework.py`) only ever saves `{"phase": ..., "step_count": ...}`,
never the `messages` list itself, and `run()` always builds `messages` fresh
from `[system, user(task)]` with no way to seed it from a prior run. So a
crash mid-run currently loses everything, despite Module 7's entire premise
being resumability.

Fix shape: checkpoint the `messages` list (or a pointer to a sidecar file
holding it) alongside phase/step_count, and give `AgentHarness` a `resume(run_id)`
path that seeds `messages` from the checkpoint instead of starting from
`[system, user(task)]`.

## High (core production gaps)

### 2. Cost/token tracking is always zero
`TraceEvent.cost_usd`/`tokens_in`/`tokens_out` (`agent_harness/tracing/ledger.py`)
default to `0`/`0.0`, and every real `TraceEvent(...)` construction in
`framework.py` omits them — so in production they're always the defaults.
The real OpenAI response carries `usage.prompt_tokens`/`completion_tokens`/
`total_tokens`, currently discarded in `OpenAIChatClient.create_action_turn`/
`create_completion`. `TraceLedger.total_cost()` sums whatever's on disk, but
for real runs that sum is always `0.0`. Module 9's whole point — cost
observability — is decorative in production today.

Fix shape: capture `response.usage` in both `OpenAIChatClient` methods,
compute cost from a small per-model pricing table, thread both through to the
`TraceEvent`s `framework.py` already logs.

### 3. Approval gate has no crash durability
`ApprovalGate` (`agent_harness/gates/approval.py`) is purely synchronous —
`check()` calls the approver inline; the default blocks on `input()`. No
`PendingAction` is ever persisted, and there's no linkage to checkpointing.
If the process crashes while waiting on approval, the pending decision is
gone. The roadmap's own Module 8 spec calls for this explicitly ("implemented
first as a blocking CLI prompt, later as an async webhook/queue") but the
"later" part was never built. Depends loosely on #1 (checkpoint resume) to be
genuinely useful — restoring a "waiting on approval" state after a crash
needs somewhere to restore it *to*.

### 4. Retry policy is too permissive
`with_backoff` (`agent_harness/persistence/retry.py`) retries everything
except `FatalError` with exponential backoff (up to 5 attempts). `FatalError`
is never raised anywhere in production code — only in a test — so a bad API
key, a malformed request, or a genuine bug surfacing as `TypeError` all get
retried 5x before finally failing, wasting time and obscuring the real error
under backoff noise.

Fix shape: in `OpenAIChatClient`, catch the OpenAI SDK's own
`AuthenticationError`/`BadRequestError`/etc. and re-raise as `FatalError` so
they fail fast instead of retrying.

## Medium (scale/quality)

### 5. Memory tiers built but unused
`DurableStateStore` and `VectorMemory` (`agent_harness/memory/`) are fully
implemented and independently tested but never imported by `framework.py` or
`run_agent.py`. `AgentHarness` has no memory-store fields; every run starts
from a blank slate beyond its own `messages` list, and nothing survives
across separate runs (user preferences, facts learned in a previous session,
semantic recall of prior work).

Fix shape: needs a product decision first — what should actually be
persisted across runs for *this* harness's use case — before it's worth
wiring in mechanically.

### 6. Orchestration not composed into a real driver
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

### 7. Prompts aren't actually versioned
`PromptTemplate`/`PromptRegistry` (`agent_harness/prompts/template.py`) exist
and are tested, but the real system/task instructions are raw f-strings
built inline in `run_agent.py` (see `RISK_BY_TOOL`'s neighboring
`instructions=(...)` block) — not routed through `PromptRegistry.render()`.
No `TraceEvent` ever records which prompt version produced a given run, so
Module 3's explicit goal (know which prompt version produced which output)
isn't actually achieved outside the module's own tests.

## Lower priority (hardening/DX)

### 8. Missing capstone integration test
`docs/agent_harness_roadmap.md`'s closing step describes
`tests/test_capstone.py`: a multi-step task through a real approval gate and
forced checkpoint, a simulated crash-and-resume via `load_checkpoint`, final
output schema validation, a gapless trace ledger with correct total cost, and
an `LLMJudge` pass across 3 deterministic runs. This file still doesn't
exist. Blocked on #1 and #2 above — the capabilities it would exercise
(checkpoint resume, non-zero cost tracking) aren't implemented yet either, so
writing this test now would just document that they're missing rather than
proving they work.

### 9. No CI coverage-threshold enforcement
`pyproject.toml`'s `[tool.pytest.ini_options].addopts` reports coverage
(`--cov=agent_harness --cov-report=term-missing`) but has no
`--cov-fail-under=N`, so `.github/workflows/ci.yml` can't fail a PR that
regresses coverage — it only fails on outright test failures.

### 10. No installable CLI / packaging
No `[project.scripts]` table in `pyproject.toml`. `run_agent.py` (invoked via
`uv run python run_agent.py "task"`) is the only entry point — no
`agent-harness run "task"` command, no config file for model/risk-tier/root
selection instead of editing the script directly.

### 11. Single-provider lock-in
`ChatClient` (`agent_harness/schemas/structured.py`) and
`ToolCallingChatClient` (`agent_harness/schemas/tool_calling.py`) are both
provider-agnostic `Protocol`s by design, but `OpenAIChatClient` is the only
implementation that exists. Nothing proves the abstraction actually holds
for a second provider, and there's no runtime provider-selection mechanism.

### 12. `TraceLedger` has no rotation/retention policy
A failure mode the roadmap itself calls out under Module 9 ("trace volume
growing unbounded... becomes an ops problem") but never addresses — `TraceLedger.log()`
just appends to one JSONL file forever.

### 13. No process-wide concurrency cap across parallel sub-agent runs
`SubAgentSpawner`'s `max_workers` (`agent_harness/orchestration/spawner.py`)
caps concurrency *within one spawner instance*, but nothing caps total
concurrent API spend if multiple spawners/harnesses run in the same process
or host — the "fork bomb of agents calling agents" failure mode the roadmap
names for Module 10. Only relevant once #6 has a real multi-agent driver to
run more than one spawner at a time.
