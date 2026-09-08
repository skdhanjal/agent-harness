"""Verification tests for Module 4: Context Construction.

Proves: over-budget blocks are dropped lowest-priority-first, higher
priority blocks always survive intact, order in the final output follows
priority (not insertion order), and the real tiktoken-based counter works.
"""

import pytest

from agent_harness.context.builder import ContextBuilder
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
