"""Verification tests for Module 4: Context Construction.

Proves: over-budget blocks are dropped lowest-priority-first, higher
priority blocks always survive intact, order in the final output follows
priority (not insertion order), the real tiktoken-based counter works, and
turn-pair compaction (the live-loop fix for unbounded `messages` growth)
summarizes whole turns without ever orphaning a `tool_call_id`.
"""

import pytest

from agent_harness.context.builder import ContextBuilder
from agent_harness.context.compaction import compact_messages
from agent_harness.context.tokens import count_tokens


def word_count(text: str) -> int:
    """Deterministic stand-in for a token counter, so tests don't depend
    on tiktoken's exact tokenization -- only on the builder's own logic."""
    return len(text.split())


def test_high_priority_block_survives_when_over_budget() -> None:
    builder = ContextBuilder(token_budget=15, count_tokens=word_count)
    builder.add("system", "one two three four five six seven eight nine ten", priority=1)
    builder.add(
        "history",
        "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
        priority=2,
    )

    output = builder.build()

    assert "one two three four five six seven eight nine ten" in output
    assert "[[history truncated: over budget]]" in output


def test_low_priority_block_truncated_before_higher_priority() -> None:
    builder = ContextBuilder(token_budget=5, count_tokens=word_count)
    builder.add("scratch", "one two three four five six seven", priority=3)
    builder.add("task", "hello world", priority=1)

    output = builder.build()

    assert "hello world" in output
    assert "[[scratch truncated: over budget]]" in output


def test_output_order_follows_priority_not_insertion_order() -> None:
    builder = ContextBuilder(token_budget=100, count_tokens=word_count)
    builder.add("history", "past turns", priority=2)
    builder.add("system", "you are an assistant", priority=1)

    output = builder.build()

    assert output.index("system") < output.index("history")


def test_all_blocks_fit_when_under_budget() -> None:
    builder = ContextBuilder(token_budget=100, count_tokens=word_count)
    builder.add("system", "you are an assistant", priority=1)
    builder.add("task", "summarize this document", priority=2)

    output = builder.build()

    assert "truncated" not in output
    assert "you are an assistant" in output
    assert "summarize this document" in output


def test_count_tokens_uses_real_tokenizer() -> None:
    try:
        baseline = count_tokens("hello world")
    except Exception as e:  # tiktoken fetches its encoding file over the network
        pytest.skip(f"tiktoken encoding unavailable, likely no network access: {e}")

    assert count_tokens("") == 0
    assert baseline > 0
    assert count_tokens("hello world, this is a longer sentence") > baseline


def _turn(n: int) -> list[dict[str, object]]:
    """One assistant+tool_result turn, padded so word_count sees real weight."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {"name": "x"}}],
        },
        {"role": "tool", "tool_call_id": f"call_{n}", "content": f"result padding {'x ' * n}"},
    ]


def test_compact_messages_returns_unchanged_when_under_budget() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    messages += _turn(1) + _turn(2)

    result = compact_messages(
        messages,
        token_budget=1000,
        count_tokens=word_count,
        summarize=lambda turns: "should not be called",
    )

    assert result is messages


def test_compact_messages_summarizes_stale_turns_and_keeps_recent_verbatim() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    turns = [_turn(n) for n in range(1, 6)]  # 5 turns, well over a tiny budget
    messages += [m for turn in turns for m in turn]

    captured: list[dict[str, object]] = []

    def fake_summarize(stale: list[dict[str, object]]) -> str:
        captured.extend(stale)
        return "condensed summary"

    result = compact_messages(
        messages,
        token_budget=5,
        count_tokens=word_count,
        summarize=fake_summarize,
        keep_recent_turns=2,
    )

    # system + task untouched
    assert result[0] == messages[0]
    assert result[1] == messages[1]
    # the 3 oldest turns (6 messages) were handed to summarize, not the 2 recent ones
    assert captured == turns[0] + turns[1] + turns[2]
    # replaced by exactly one summary message
    assert result[2] == {
        "role": "user",
        "content": "[Summary of 3 earlier turns]\ncondensed summary",
    }
    # the 2 most recent turns survive verbatim, tool_call_id pairing intact
    assert result[3:] == turns[3] + turns[4]


def test_compact_messages_leaves_single_oversized_turn_alone() -> None:
    """Can't shrink what it can't split: with <= keep_recent_turns turns total,
    there's nothing older to summarize away, even over budget."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    messages += _turn(1)

    result = compact_messages(
        messages,
        token_budget=1,
        count_tokens=word_count,
        summarize=lambda turns: "should not be called",
        keep_recent_turns=2,
    )

    assert result is messages
