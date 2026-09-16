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

### Approval gate has no crash durability — fixed
`ApprovalGate.check()` (`agent_harness/gates/approval.py`) now persists every
gated decision by `action.id` (the provider's own tool_call_id) through a new
`agent_harness/gates/pending_store.py` — `status="pending"` is written to
disk *before* the approver is invoked, and the resolution
(`"approved"`/`"denied"`) right after. A repeat `check()` for an id that
already resolved returns that decision without touching the approver again.
On its own this only gave crash *evidence*, not crash *recovery* — the real
gap was that `AgentHarness._run_loop` (`agent_harness/framework.py`) only
checkpointed at whole-iteration boundaries, so a crash mid-batch (e.g.
blocked approving the 2nd of 3 tool calls) had no checkpointed record of
which calls the model had already decided on or which had already run.
`_run_loop` now checkpoints after the assistant's tool-call message and
after each individual tool call, and a new `_unresolved_tool_calls` scan
lets both fresh entry and `resume()` detect an in-progress, not-fully-
answered tool-call batch and finish exactly the unresolved calls instead of
re-calling the model. Full design rationale — why `tool_call_id` doubles as
the durable id instead of minting a new one, why the pending store is a
sibling of `persistence/checkpoint.py` rather than folded into it, and why
persisting the decision alone wasn't enough without the finer checkpoint
granularity — is written up in `docs/agent_harness_roadmap.md`'s Module 8
section. Tested in `tests/test_module_08_approval_gates.py` (decision
persisted before the approver runs; a resumed gate replays an already-made
approval/denial without re-asking) and `tests/test_framework.py`
(`test_harness_resumes_mid_tool_call_batch_without_reexecuting_or_reasking`).

### Durable facts memory built but unused — fixed
`DurableStateStore` (`agent_harness/memory/durable_store.py`) is now wired
into `run_agent.py` via a new `agent_harness/memory/tools.py` —
`register_memory_tools()` exposes it to the model as `remember_fact`,
`recall_fact`, and `list_facts`, the same way `register_filesystem_tools()`
exposes the sandboxed filesystem. Unlike every other run artifact
(checkpoints, pending actions, traces), the store lives at one fixed path
shared across `run_id`s (`run_agent.py`'s `MEMORY_PATH`), not a per-run one
— a fact now genuinely survives the process that wrote it. Full design
rationale — why this is tool exposure rather than an `AgentHarness`
constructor field or automatic context injection, why the store is
cross-run rather than per-run, and why `remember_fact` sits at
`RiskTier.MEDIUM` rather than `HIGH` — is written up in
`docs/agent_harness_roadmap.md`'s Module 6 section. Tested in
`tests/test_memory_tools.py` (tools round-trip through the registry's
normal validate-execute path, and a fact survives a fresh
`DurableStateStore` instance pointed at the same file).

`VectorMemory` (semantic retrieval) is deliberately **not** part of this
fix — see "Semantic memory (VectorMemory) still unwired" below, now split
out as its own item since it turned out to need a materially different kind
of decision (an embeddings provider and a persistence format that don't
exist yet, not just a wiring change).

### Orchestration not composed into a real driver — fixed
`agent_harness/orchestration/driver.py`'s new `OrchestrationDriver` is the
piece that actually composes `ModelRouter` and `SubAgentSpawner`: it picks
each `DagNode`'s starting tier from `node.difficulty`, runs the pending
batch through `SubAgentSpawner`, and retries any node that failed one tier
up via `ModelRouter.escalate()`, up to `max_attempts` — nodes that already
succeeded are never re-run. `run_orchestrated.py` (new, the multi-agent
counterpart to `run_agent.py`) wires this against real `AgentHarness`
instances, each with its own `run_id`, trace file, and checkpoint dir
nested under the parent orchestration run. Getting there also required
fixing `SubAgentSpawner` itself: `run_dag` no longer lets one node's
exception propagate and abort the whole batch (`NodeResult(ok, result,
error)` now, mirroring `ToolExecutionResult`), and `harness_factory` now
receives the node being built for, not zero args — neither was survivable
by a driver that needs to see per-node failures and give different nodes
different config. Full design rationale — including why `run_id` ended up
*not* added to `AgentHarness.run()`, a deliberate change from this item's
original fix-shape note — is written up in
`docs/agent_harness_roadmap.md`'s Module 10 section. Tested in
`tests/test_module_10_orchestration.py` (router behavior, spawner
completing all nodes under `max_workers` while capturing a failing node's
error without losing a sibling's result) and
`tests/test_orchestration_driver.py` (tier-by-difficulty, escalate-and-
retry on failure, give-up after `max_attempts`, one node's failure never
blocking a sibling).

`docs/agent_harness_roadmap.md`'s own spawner sketch used a single reused
harness instance and a `run_id` passed to `run()` — deliberately not what
got built; see Module 10's "why this design" for why a fresh harness per
node ended up the right call instead. LLM-driven task decomposition
(turning one task string into this `DagNode` list) is deliberately **not**
part of this fix — see "LLM-driven task decomposition not implemented"
below, split out as its own item since it's a planning feature with its
own failure modes, not a wiring change.

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

### Prompts aren't actually versioned — fixed
`run_agent.py` now registers its instructions as `filesystem_agent@v1` in a
module-level `PromptRegistry` and renders both the system and user halves
through `PromptTemplate.render()` instead of building an f-string inline.
`AgentHarness` (`agent_harness/framework.py`) takes the resulting
`RenderedPrompt.prompt_id` as a new constructor field and logs it once, in
a `run_start` `TraceEvent`, at the top of `run()` — so which prompt version
produced a given run's output is now recoverable from the trace ledger
itself instead of being lost outside `run_agent.py`. Full design
rationale — why `prompt_id` is logged once per run instead of on every
`llm_call` event, why prompt selection stays a caller concern rather than
moving into `AgentHarness`, and why the user-half template is a trivial
`"$task"` passthrough — is written up in `docs/agent_harness_roadmap.md`'s
Module 3 section. Tested in `tests/test_module_03_prompt_ownership.py` and
`tests/test_framework.py::test_harness_logs_prompt_id_at_run_start`.

### LLM-driven task decomposition not implemented — fixed
New `agent_harness/orchestration/decomposition.py`'s `decompose_task(task,
client)` turns one task string into a `list[DagNode]` via
`generate_structured` (Module 1), reusing the `ChatClient` protocol the
same way `LLMJudge` already does. It rejects and re-prompts on an empty
decomposition, duplicate ids, a `depends_on` referencing an unknown id, a
cyclic dependency graph, and — the check that actually matters given
`SubAgentSpawner.run_dag`'s real capability — any non-empty `depends_on`
at all, since nodes always run fully in parallel. `run_orchestrated.py`
now takes a task string (`sys.argv`, same pattern as `run_agent.py`) and
calls `decompose_task` instead of using a hand-written `DEFAULT_NODES`.
Full design rationale — why `depends_on` stays in the schema as a reject
condition rather than being dropped or partially supported, why the
business-rule retry loop is separate from `generate_structured`'s own
schema-repair loop, why decomposition is its own module rather than folded
into `driver.py` — is written up in `docs/agent_harness_roadmap.md`'s
Module 10 section. Tested in `tests/test_task_decomposition.py`.

### Missing capstone integration test — fixed
New `tests/test_capstone.py` exercises `AgentHarness`, `ApprovalGate`,
checkpointing/resume, `TraceLedger`, and `LLMJudge` together in one
continuous run instead of each module's own isolated test file:
`test_capstone_survives_crash_and_resume_with_gapless_trace_and_valid_output`
runs a two-tool-call batch through a real `ApprovalGate`, forces a
mid-batch crash (`FatalError` while approving the second call, same
technique as `tests/test_framework.py`'s mid-batch resume test), asserts
the resulting checkpoint captured exactly the first call's result, resumes
into a fresh `AgentHarness`, validates the resumed run's final answer
against a local `RunSummary` pydantic schema, and asserts the trace ledger
has exactly the expected `run_start`/`llm_call`/`tool_call` event counts
and total cost across both the pre-crash and post-resume turns.
`test_capstone_llm_judge_scores_three_deterministic_runs_consistently`
runs the same scripted scenario 3 times, confirms identical output each
time, and scores each run's trajectory with `LLMJudge`. No production code
changed — this was purely a missing-test gap, not a wiring gap like the
other items above.

## Medium (scale/quality)

### 1. Semantic memory (VectorMemory) still unwired
`VectorMemory` (`agent_harness/memory/vector_memory.py`) is fully
implemented and independently tested but never imported by `framework.py`
or `run_agent.py` — facts memory (`DurableStateStore`) is fixed (see
above), but semantic recall of prior work (e.g. "what did I do last time on
a task like this") still isn't available. This is a materially bigger lift
than the facts-memory fix was: `VectorMemory` has no disk persistence at
all today (in-memory `_texts`/`_vectors` lists, gone at process exit), and
no embedding provider exists anywhere in the codebase —
`OpenAIChatClient` (`agent_harness/schemas/openai_client.py`) has no
`embed()` method, only chat completions.

Fix shape: needs its own product decision — real OpenAI embeddings
(`text-embedding-3-small`, a second billed API surface) vs. a lightweight
local embedder (weaker retrieval quality, zero extra cost/dependency) —
plus a persistence format for the vectors, before it's a well-scoped
wiring change like the facts-memory fix was.

## Lower priority (hardening/DX)

### 2. No CI coverage-threshold enforcement
`pyproject.toml`'s `[tool.pytest.ini_options].addopts` reports coverage
(`--cov=agent_harness --cov-report=term-missing`) but has no
`--cov-fail-under=N`, so `.github/workflows/ci.yml` can't fail a PR that
regresses coverage — it only fails on outright test failures.

### 3. No installable CLI / packaging
No `[project.scripts]` table in `pyproject.toml`. `run_agent.py` (invoked via
`uv run python run_agent.py "task"`) is the only entry point — no
`agent-harness run "task"` command, no config file for model/risk-tier/root
selection instead of editing the script directly.

### 4. Single-provider lock-in
`ChatClient` (`agent_harness/schemas/structured.py`) and
`ToolCallingChatClient` (`agent_harness/schemas/tool_calling.py`) are both
provider-agnostic `Protocol`s by design, but `OpenAIChatClient` is the only
implementation that exists. Nothing proves the abstraction actually holds
for a second provider, and there's no runtime provider-selection mechanism.

### 5. `TraceLedger` has no rotation/retention policy
A failure mode the roadmap itself calls out under Module 9 ("trace volume
growing unbounded... becomes an ops problem") but never addresses — `TraceLedger.log()`
just appends to one JSONL file forever.

### 6. No process-wide concurrency cap across parallel sub-agent runs
`SubAgentSpawner`'s `max_workers` (`agent_harness/orchestration/spawner.py`)
caps concurrency *within one spawner instance*, but nothing caps total
concurrent API spend if multiple spawners/harnesses run in the same process
or host — the "fork bomb of agents calling agents" failure mode the roadmap
names for Module 10. `OrchestrationDriver` (see "Fixed" above) makes this
easier to hit in practice, since a single `run_orchestrated.py` invocation
can now genuinely spin up several concurrent `AgentHarness` instances; there
still isn't anything capping spend *across* separate `OrchestrationDriver`
runs in the same process or host.
