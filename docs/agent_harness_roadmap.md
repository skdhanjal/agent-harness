# Agent Harness — Architecture

A from-scratch, production-grade agent harness built across 10 core
Agent-Harness Design Patterns: structured outputs, tool calling, prompt
ownership, context construction, control-flow ownership, state & memory,
checkpointing & retries, permission & approval gates, tracing & evals, and
adaptive orchestration.

The organizing principle across all ten: **the harness owns control flow,
the model only proposes.** Every decision the model makes — call a tool,
declare itself done, transition state — passes through a harness-owned
module that validates, gates, logs, or rejects it before it takes effect.
No LangChain/LlamaIndex/AutoGen — raw OpenAI API calls + `pydantic`, so the
mechanics are explicit rather than hidden behind a framework.

This document describes the system as it actually works today. For what's
still missing to call it truly production-ready, see
`docs/production_readiness_todo.md`.

## How a task flows through the harness

Everything below is one loop, driven by `AgentHarness.run()`
(`agent_harness/framework.py`). The model is a normal multi-turn chat
participant: each turn it either requests tool calls (validated by the
provider against each tool's own schema) or replies with plain text, which
the harness treats as the final answer.

```mermaid
flowchart TD
    Start([Task submitted]) --> Seed["Seed messages:\nsystem = instructions, user = task"]
    Seed --> Guard{"iterations &lt;\nmax_iterations?"}
    Guard -- no --> Fail([raise MaxIterationsExceededError])
    Guard -- yes --> Call["Call the model\n(native tool-calling turn)"]
    Call --> Decide{"Tool calls\nrequested?"}
    Decide -- "no (plain text)" --> Done["FSM: EXECUTE → REVIEW → DONE"]
    Done --> Return([Return final answer])
    Decide -- yes --> Assistant["Append assistant message\n(with tool_calls)"]
    Assistant --> PerCall["For each requested tool call"]
    PerCall --> Risk{"Risk tier at/above\ngate threshold?"}
    Risk -- below --> Exec["Execute via ToolRegistry:\nvalidate args, resolve through\nPathSandbox, run with timeout"]
    Risk -- "at/above" --> Approve{"Approver\napproves?"}
    Approve -- yes --> Exec
    Approve -- no --> Deny["Append DENIED tool result"]
    Exec --> Result["Append tool result\n(role=tool, tool_call_id)"]
    Deny --> Compact{"messages over\ntoken budget?"}
    Result --> Compact
    Compact -- no --> Loop["FSM: → EXECUTE"]
    Compact -- "yes, and >\nkeep_recent_turns" --> Summarize["Summarize oldest turn(s)\ninto one message"]
    Summarize --> Loop
    Loop --> Guard

    Call -.-> Trace[("TraceLedger:\nllm_call / tool_call /\ngate_decision /\ncontext_compaction events")]
    Loop -.-> Checkpoint[("Checkpoint:\nphase + step_count + messages")]
```

A few things worth reading directly off this diagram:

- **The `messages` list is the only source of truth.** It's seeded once
  (`system` = harness instructions, `user` = the task) and grows for the
  whole run — the model's own tool-call request and the tool's result are
  both appended, linked by `tool_call_id`. There's no separate paraphrased
  history to keep in sync with it.
- **"Final answer" is structural, not self-reported.** The model never says
  "I'm done" in a field the harness has to trust — the absence of
  `tool_calls` in the response *is* the done signal.
- **Every tool call passes through the same gate**, regardless of which
  tool it is: look up its `RiskTier`, and if that's at or above the
  configured threshold, block on `ApprovalGate` before `ToolRegistry.execute`
  ever runs.
- **Tracing and checkpointing are side effects of the loop, not steps in
  it** — they attach after every model call and after every state
  transition, not conditionally.
- **Context compaction is a budget check on every iteration, not a
  reaction to failure.** The loop never waits for a context-length error
  from the API — it counts tokens itself and condenses old turns away
  before that could happen.
- **The step budget is a hard ceiling**, not a suggestion: `AgentFSM`'s own
  `max_steps` guard (separate from `max_iterations` here) raises if the FSM
  itself is driven past its limit.

## Package layout

```
agent_harness/
├── framework.py         # AgentHarness -- the loop diagrammed above
├── schemas/
│   ├── structured.py     # generate_structured(): text -> validated pydantic, with repair
│   ├── actions.py         # AgentAction/ActionType -- superseded for tool selection, see below
│   ├── tool_calling.py     # ToolCallingChatClient protocol, ChatTurn, ToolCallRequest
│   └── openai_client.py     # the only file that imports the `openai` SDK
├── tools/
│   ├── registry.py        # ToolRegistry: schema derivation, arg validation, sandboxed execution
│   ├── filesystem.py       # read/write/list/glob/grep/stat/mkdir/move/delete
│   └── sandbox.py           # PathSandbox: confines every tool path to a root
├── prompts/template.py    # PromptTemplate/PromptRegistry -- built, not yet wired in (see gaps)
├── context/
│   ├── builder.py           # ContextBuilder -- block truncation, built, not yet wired in (see gaps)
│   ├── tokens.py             # count_tokens, count_messages_tokens
│   └── compaction.py          # compact_messages: wired into the live loop, see Module 4 below
├── control_flow/state_machine.py   # AgentFSM
├── memory/
│   ├── sliding_window.py    # no longer used by the live loop (see Module 6 below)
│   ├── durable_store.py      # built, not yet wired in (see gaps)
│   └── vector_memory.py       # built, not yet wired in (see gaps)
├── persistence/
│   ├── checkpoint.py        # save_checkpoint/load_checkpoint
│   └── retry.py               # with_backoff, FatalError
├── gates/approval.py       # RiskTier, ApprovalGate, PendingAction
├── tracing/
│   ├── ledger.py             # TraceLedger, TraceEvent
│   └── judge.py                # LLMJudge
└── orchestration/
    ├── router.py             # ModelRouter -- built, not yet composed into a driver (see gaps)
    └── spawner.py              # SubAgentSpawner -- same
```

## Module reference

### 1. Structured Outputs — `schemas/structured.py`, `schemas/actions.py`

`generate_structured()` turns raw model text into a validated Pydantic
object: it asks for JSON matching a schema, and on `ValidationError` feeds
the exact error back to the model in a bounded repair loop rather than
failing on the first malformed response. This was originally also how the
agent loop picked tools — the model would emit an `AgentAction` JSON object
with a `tool_name`/`tool_args` field — but that's been superseded by native
tool-calling (see the flow diagram and the design note below). It's still
the general-purpose mechanism for other structured needs, e.g. `LLMJudge`'s
numeric score in Module 9.

Tested in `tests/test_module_01_structured_outputs.py`.

### 2. Tool Calling — `tools/registry.py`, `tools/filesystem.py`, `tools/sandbox.py`

`ToolRegistry.tool` derives a Pydantic argument model from a plain
function's type hints — no hand-written JSON schema to drift out of sync.
`execute()` validates the model-supplied args against it, runs the function
in a thread pool with a timeout, and always returns a
`ToolExecutionResult(ok, result, error)` — a tool can never raise across
this boundary. The filesystem tools (`read_file`, `write_file`,
`list_files`, `glob_files`, `grep_files`, `file_stat`, `make_directory`,
`move_file`, `delete_file`) resolve every path through a `PathSandbox`
first, which normalizes the path with `Path.resolve()` (collapsing `..` and
following symlinks in one step) and rejects anything that lands outside the
configured root.

Tested in `tests/test_module_02_tool_calling.py`, `tests/test_filesystem_tools.py`.

### 3. Prompt Ownership — `prompts/template.py`

`PromptTemplate`/`PromptRegistry` give prompts a name, a version, and strict
variable substitution (a missing variable fails loudly, naming the exact
prompt and version, instead of silently blanking). **Not currently wired
in** — the harness's real instructions live as raw f-strings in
`run_agent.py`. See `docs/production_readiness_todo.md`.

Tested in `tests/test_module_03_prompt_ownership.py`.

### 4. Context Construction — `context/builder.py`, `context/tokens.py`, `context/compaction.py`

`ContextBuilder` assembles prioritized text blocks into one budgeted
string, dropping the lowest-priority blocks whole (never mid-sentence) once
a token budget is exceeded. It's still **not wired into the live loop** —
it can't be, as-is: the native tool-calling `messages` list has a
structural constraint independent named blocks don't, an `assistant`
message with `tool_calls` must be immediately followed by one `tool`
message per call. Dropping one without the other produces a message list
the API rejects.

`compact_messages` (`context/compaction.py`) is the loop-specific fix, and
*is* wired in. `AgentHarness._compact_if_needed` checks
`count_messages_tokens(messages)` against `context_token_budget` once per
iteration, right after that iteration's assistant message and tool results
are appended. Once over budget, it always moves or summarizes a whole
`(assistant, *tool_results)` turn at a time — never a lone message — so a
`tool_call_id` is never left orphaned. The system message and original
task are kept forever; the most recent `keep_recent_turns` (default 2)
turns are kept verbatim as the model's working context; everything older
is condensed into one message by a summarization call that reuses the
harness's own `client.create_action_turn(prompt, tools=[])` — no second
LLM client dependency. Compaction is logged as a `context_compaction`
`TraceEvent` (`tokens_before`/`tokens_after`), so it's observable, not
silent.

**Why this design, and what else was considered:**

- **Why the unit of compaction is a whole turn, not a message.** This is
  the load-bearing constraint, covered above: the API rejects an
  `assistant` message whose `tool_calls[].id` has no matching `tool` reply.
  Any scheme that could drop or move a `tool` message independently of its
  `assistant` message is disqualified before it's even a design choice.
- **Why summarize instead of just dropping old turns outright.** Dropping
  is simpler and cheaper (no LLM call), but it silently destroys
  information the model may still need — a fact discovered three tool
  calls ago, a decision already made. Summarizing costs one extra LLM call
  per compaction event but keeps a compressed trace of what happened,
  which matches how a person would compress their own memory of a long
  task rather than just forgetting the first half of it.
- **Why reuse `self.client` for summarization instead of injecting a
  second `ChatClient`.** The alternative — add a `summarizer: ChatClient`
  constructor param, matching Module 1's older `create_completion` protocol
  — would add a second injectable dependency and a second Protocol for
  every caller to wire up, just to get plain text back. `create_action_turn`
  already returns plain text when called with `tools=[]`
  (`openai_client.py`'s `create_action_turn` skips the `tools=` kwarg
  entirely in that case), so the existing dependency does the job. One
  Protocol, one client, less to configure.
- **Why the summary is inserted as `role: user`, not `role: system` or
  `role: assistant`.** `assistant` would misattribute authorship — the
  model never actually said this, and a later confused turn referring back
  to "what I said" would be reasoning from a fabricated self-quote.
  `system` is reserved for the one instructions message seeded at the top
  of `messages`; overloading it mid-conversation blurs the one message the
  model is trained to weight most heavily as its operating instructions.
  `user` reads as exactly what it is: contextual information supplied to
  the model, not by it.
- **Why the check runs after appending a turn's messages, not before the
  next model call.** Both would eventually catch an over-budget list, but
  checking post-append keeps the boundary turn-aligned for free — the
  turn that just completed is either entirely in or entirely candidate for
  summarization, never half-appended when the check runs.
- **Why `context_token_budget` defaults to 8,000.** Deliberately
  conservative relative to real model context windows (gpt-4o-mini is
  128K) — the goal is to exercise compaction well before a run is ever at
  real risk of hitting the model's actual limit, not to squeeze maximum
  context out of every call. It's a constructor parameter specifically so
  a caller can raise it for a model with a smaller window or a task that
  needs more working history verbatim.
- **What was rejected: reusing `ContextBuilder` directly.** Considered and
  dropped immediately — see the constraint above. `ContextBuilder` remains
  a legitimate tool for a different problem (assembling one bounded prompt
  from independent named sections that have no pairing constraint between
  them), just not this one.

**Known limitation, not solved here:** if the kept recent turns alone
exceed the budget (e.g. one huge tool result), this pass can't shrink
them — turn-pair compaction only ever helps by summarizing *older* turns
away. Fixing that is a different problem (per-message truncation) than
this module solves.

Tested in `tests/test_module_04_context_construction.py` (pure
`compact_messages` unit tests) and `tests/test_framework.py`
(`test_harness_compacts_context_once_over_budget`, an end-to-end run that
forces compaction mid-loop).

### 5. Control-Flow Ownership — `control_flow/state_machine.py`

`AgentFSM` enforces the legal phase graph —
`PLAN → EXECUTE → {REVIEW, EXECUTE} → {EXECUTE, DONE, FAILED}` — plus a hard
`max_steps` guard independent of what the model wants to do. The model's
output is a *proposal*; `AgentFSM.step()` is the only thing that actually
advances state, and raises on an illegal transition or an exceeded step
budget rather than allowing it silently.

Tested in `tests/test_module_05_control_flow.py`.

### 6. State and Memory — `memory/`

`SlidingWindowMemory` is no longer used by the live loop: once tool calls
became native, the real `messages` list already carries full turn-by-turn
history (tool requests and results linked by `tool_call_id`), which made
the separate paraphrased memory log redundant — keeping both in sync would
have been pure risk for no benefit. `DurableStateStore` (crash-durable
key-value facts) and `VectorMemory` (semantic retrieval) are both fully
built and independently tested but not wired into the harness at all yet.
See `docs/production_readiness_todo.md`.

Tested in `tests/test_module_06_memory.py`.

### 7. Checkpointing & Safe Retries — `persistence/`

`with_backoff()` wraps every model call with exponential backoff + jitter,
re-raising immediately on `FatalError` instead of retrying it — though
nothing in production code raises `FatalError` today, so a bad API key
currently gets retried like any transient failure. See
`docs/production_readiness_todo.md`.

`save_checkpoint()` (via `AgentHarness._checkpoint`) now writes `{phase,
step_count, messages}` after every step in the diagram above, and
`AgentHarness.resume()` is the counterpart that actually uses
`load_checkpoint()`: it loads that state for `self.run_id`, restores
`AgentFSM.phase`/`step_count` by direct field assignment (the FSM is a
plain mutable dataclass, `control_flow/state_machine.py:43`), and re-enters
the same `_run_loop` that `run()` uses — one loop implementation, two ways
to seed it. Two calls it can raise instead of silently doing the wrong
thing: `CheckpointNotFoundError` (no checkpoint for this `run_id`) and
`RunAlreadyFinishedError` (checkpoint's phase is already `DONE`/`FAILED` —
Module 5's own terminal states, so there's nothing left to continue).

**Why this design:**

- **`messages` lives inline in the checkpoint JSON, not a sidecar file.**
  One file, one write, no two files to keep in sync. This only stays cheap
  because Module 4's compaction (above) already bounds how large `messages`
  can grow — the two fixes compose.
- **`resume()` takes no arguments and always reads from disk, never trusts
  `self.fsm` in memory.** The real caller is a brand-new `AgentHarness` in a
  new process after a crash — its own `self.fsm` starts at `PLAN`/`0` and
  has never seen the pre-crash state, so there's nothing in memory worth
  trusting.
- **One shared `_run_loop`, not two copies of the model-call loop.** `run()`
  seeds `messages` from `[system, user(task)]` and does the mandatory
  `PLAN → EXECUTE` transition; `resume()` seeds `messages` and FSM state
  from the checkpoint instead. Both then call the same loop. Two copies of
  the loop body would drift the moment one of them changed without the
  other.
- **No migration path for checkpoints written before this change.**
  Checkpoints are ephemeral per-run scratch state, not a versioned public
  format — an old checkpoint missing the `"messages"` key just can't be
  resumed, and nothing tries to paper over that.
- **The resumed segment gets a fresh `max_iterations` model-call budget,
  but the FSM's `step_count` ceiling (restored from the checkpoint) is the
  real cross-process hard limit.** If a run crashes after 3 of 5
  iterations and resumes, the resumed segment can make up to 5 more model
  calls — but `AgentFSM`'s own `max_steps` guard, continuing from the
  pre-crash `step_count`, still raises `StepLimitExceededError` once the
  *original* total ceiling is hit. Two guards, two different jobs: one
  paces a single process's model calls, the other is the absolute ceiling
  that survives a crash.

Tested in `tests/test_module_07_checkpointing.py` (checkpoint round-trip
in isolation) and `tests/test_framework.py`
(`test_harness_resumes_from_checkpoint_after_crash` and the two
`test_resume_raises_*` tests for the not-found / already-finished cases).

### 8. Permission & Approval Gates — `gates/approval.py`

`RiskTier` (`LOW < MEDIUM < HIGH`) is assigned per tool name at harness
construction (`risk_by_tool` in `run_agent.py`). `ApprovalGate.check()`
auto-approves anything below `require_approval_at` and defers everything
else to an `approver` callback — a blocking CLI prompt by default, an
auto-approve function for non-interactive runs. This is the gate/approve
branch in the flow diagram above.

Tested in `tests/test_module_08_approval_gates.py`.

### 9. Tracing & Evals — `tracing/`

`TraceLedger` appends one JSON line per `llm_call`/`tool_call`/
`gate_decision`/`context_compaction` event, and `total_cost()` sums
`cost_usd` across a run's events. `ChatTurn` (`schemas/tool_calling.py`)
carries `tokens_in`/`tokens_out`/`cost_usd`; `OpenAIChatClient.create_action_turn`
populates them from the real `response.usage` and a small per-model USD
price table, and `AgentHarness._run_loop` copies all three onto every
`llm_call` `TraceEvent` it logs — so `total_cost()` now sums real numbers
for a real run instead of always `0.0`. `LLMJudge` scores a finished
trajectory against a rubric, reusing `generate_structured` from Module 1.

**Why this design:**

- **The price table lives in `openai_client.py`, not a shared `tracing`
  module.** Per-token USD pricing is a fact about the *provider*, not about
  tracing in general — the same reason `openai_client.py` is the only file
  that imports the `openai` SDK. Putting it in `tracing/` would also invert
  the module dependency graph above (M1 → M9, never the reverse); keeping it
  in `schemas/openai_client.py` means M9 stays a pure consumer of whatever
  numbers `ChatTurn` already carries, with zero awareness of how they were
  priced.
- **Unrecognized models fall back to `gpt-4o-mini`'s rate instead of
  raising.** A live run's cost estimate being slightly off for a brand-new
  model is a rounding error; raising mid-run over a pricing-table miss would
  turn an observability gap into an availability one, which is a worse
  trade.
- **`create_completion` (the older Module 1 `ChatClient` protocol, used by
  `generate_structured` and `LLMJudge`) deliberately wasn't touched.** Its
  contract returns a plain `str`, and neither of its callers ever logs a
  `TraceEvent` — there's nowhere for a `tokens_in`/`cost_usd` it captured to
  actually go yet. Widening that Protocol's return type would ripple through
  every caller and test fake for a value nothing downstream reads. If
  Module 1 call sites ever get their own tracing, that's the point to
  revisit this, not before.

Tested in `tests/test_openai_client.py` (usage capture and cost math against
a mocked SDK response, including the missing-`usage` and unpriced-model
fallbacks) and `tests/test_module_09_tracing_evals.py` (ledger/judge unit
tests) and `tests/test_framework.py`
(`test_harness_records_token_usage_and_cost_from_chat_turn`, proving the
harness threads a turn's numbers through to the ledger unchanged).

### 10. Adaptive Orchestration — `orchestration/`

`ModelRouter` picks a model tier from estimated difficulty and prior
failure count, capped at the top tier no matter how many failures pile up.
`SubAgentSpawner` runs a DAG of sub-tasks across a bounded thread pool, each
as its own harness instance. Neither is composed into a real multi-agent
driver yet — `AgentHarness.run()` doesn't even accept a `run_id` override,
which a real spawner would need to give each sub-agent its own trace file.
See `docs/production_readiness_todo.md`.

Tested in `tests/test_module_10_orchestration.py`.

## Design decisions worth calling out

**Native tool-calling over a hand-rolled JSON action schema.** The harness
originally had the model emit one JSON object (`reasoning`, `action_type`,
`tool_name`, `tool_args`) validated against a schema — Module 1's repair
loop, repurposed for control flow. That broke down once the model also
needed to know what tools existed: exposing tool schemas via OpenAI's
native `tools=` parameter *while also* forcing `response_format=json_object`
made the model drift into native-tool-calling artifacts inside the JSON
text (a `tool_name` prefixed `"functions."`, and once an `action_type`
value borrowed from OpenAI's own parallel-tool-call format that isn't even
in the schema) — and kept it re-invoking a tool even after the task was
already done, since nothing in that mixed signal ever cleanly meant "stop."
Switching to the provider's real tool-calling collapses this into two clean
channels: a `tool_calls` response the API itself validates per-tool, or
plain text treated as the final answer.

**Filesystem tools are sandboxed by construction.** A single containment
check (`resolved.is_relative_to(root)`, after `Path.resolve()`) catches both
a `../` traversal and a symlink planted inside the root that points
outside it, because `resolve()` normalizes both in one step. No tool needs
its own escape handling — `PathSandbox.resolve()` raises, and
`ToolRegistry.execute`'s blanket exception handling turns that into a
structured `ok=False` result like any other tool failure.

## Module dependency graph

Build order was roughly 1→10, but the real dependencies are narrower than a
strict chain:

```mermaid
flowchart LR
    M1[1. Structured Outputs] --> M2[2. Tool Calling]
    M1 --> M3[3. Prompt Ownership]
    M2 --> M4[4. Context Construction]
    M3 --> M4
    M4 --> M5[5. Control-Flow Ownership]
    M5 --> M6[6. State and Memory]
    M6 --> M7[7. Checkpointing and Retries]
    M2 --> M8[8. Permission and Approval Gates]
    M5 --> M8
    M1 --> M9[9. Tracing and Evals]
    M3 --> M9
    M7 --> M10[10. Adaptive Orchestration]
    M8 --> M10
    M9 --> M10
```

1-3 are the typed I/O contracts (structured outputs, tool safety, versioned
prompts) everything else assumes. 4-5 turn those into an actual reasoning
loop with a context budget and deterministic control. 6-7 make the loop
stateful and crash-resilient. 8-9 add safety and observability, both only
meaningful once a real loop exists to gate or watch. 10 is the capstone:
it composes the whole harness as the unit it routes between model tiers and
spawns copies of.

## Known gaps

Every module above passes its own test in isolation — that's not the same
as the whole system being safe to run unattended against real work.
`docs/production_readiness_todo.md` tracks that gap in full, prioritized
detail; see it for what's still open (approval-gate crash durability, retry
policy, and further down the priority list).

(Unbounded context growth, checkpoint resume, and cost/token tracking were
the most load-bearing items in this list; all three are now fixed — see
Module 4, Module 7, and Module 9 above.)
