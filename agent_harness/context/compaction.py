"""Turn-pair compaction for the live loop's `messages` list.

Compacts whole (assistant, *tool_results) turns, never a lone message, so
a tool_call_id is never orphaned -- unlike `ContextBuilder` (builder.py),
which truncates independent blocks with no such pairing constraint. Full
design rationale: docs/agent_harness_roadmap.md, Module 4.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_harness.context.tokens import count_messages_tokens


def _split_turns(messages: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Group messages into (assistant, *tool_results) turns.

    Assumes the leading system/user pair has already been stripped.
    """
    turns: list[list[dict[str, object]]] = []
    for message in messages:
        if message.get("role") == "assistant":
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def compact_messages(
    messages: list[dict[str, object]],
    token_budget: int,
    count_tokens: Callable[[str], int],
    summarize: Callable[[list[dict[str, object]]], str],
    keep_recent_turns: int = 2,
) -> list[dict[str, object]]:
    """Summarize the oldest turns into one message once over budget.

    Keeps `messages[:2]` (system + task) and the last `keep_recent_turns`
    turns verbatim; everything older is replaced by one `summarize` call.
    Returns `messages` unchanged if already under budget, or if there's
    nothing older than `keep_recent_turns` to compact away.
    """
    if count_messages_tokens(messages, count_tokens) <= token_budget:
        return messages

    head, rest = messages[:2], messages[2:]
    turns = _split_turns(rest)
    if len(turns) <= keep_recent_turns:
        return messages

    stale_turns, recent_turns = turns[:-keep_recent_turns], turns[-keep_recent_turns:]
    stale_messages = [message for turn in stale_turns for message in turn]
    recent_messages = [message for turn in recent_turns for message in turn]

    summary = summarize(stale_messages)
    summary_message: dict[str, object] = {
        "role": "user",
        "content": f"[Summary of {len(stale_turns)} earlier turns]\n{summary}",
    }
    return [*head, summary_message, *recent_messages]
