# agent-harness

agent-harness built from ground up

A from-scratch, production-grade agent harness, built module by module across
the 10 core Agent-Harness Design Patterns: structured outputs, tool calling,
prompt ownership, context construction, control-flow ownership, state &
memory, checkpointing & retries, permission & approval gates, tracing &
evals, and adaptive orchestration.

No LangChain/LlamaIndex/AutoGen — raw API calls + `pydantic`, until the
mechanics are understood by hand.

## Setup

```bash
uv sync --all-groups        # installs runtime + dev dependencies
uv run pre-commit install   # enforces lint/format/type checks before each commit
```

## Day-to-day commands

```bash
uv run pytest                    # run the test suite
uv run ruff check . --fix        # lint
uv run ruff format .             # format
uv run mypy agent_harness        # type-check
```

CI (`.github/workflows/ci.yml`) runs all four on every push to `main`.

## Workflow

- Work happens module by module, straight on `main` — each module is
  committed once its Verification Test passes.
- Commit style: `feat(m<N>): <what>` / `test(m<N>): <what>`, e.g.
  `feat(m1): add structured-output repair loop`.
- `main` stays green: if CI is red, the next commit fixes it before
  anything else lands.
- After each module's tests pass, tag the commit: `git tag v0.<N>-module<N>`
  — a rollback point per pattern, mirroring Module 7's own checkpointing.

## Module tracker

| # | Pattern | Status |
|---|---|---|
| 1 | Structured Outputs | ✅ done |
| 2 | Tool Calling | ✅ done |
| 3 | Prompt Ownership | ✅ done |
| 4 | Context Construction | ✅ done |
| 5 | Control-Flow Ownership | ✅ done |
| 6 | State and Memory | ✅ done |
| 7 | Checkpointing & Safe Retries | ✅ done |
| 8 | Permission & Approval Gates | ✅ done |
| 9 | Tracing & Evals | ✅ done |
| 10 | Adaptive Orchestration | ⬜ not started |

See `docs/agent_harness_roadmap.md` for the full architecture, per-module
concepts, boilerplate, failure modes, and verification tests.
