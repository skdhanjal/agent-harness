"""Verification tests for Module 7: Checkpointing & Safe Retries.

Proves: with_backoff retries transient failures then succeeds, gives up
after max_retries, never retries a FatalError; checkpoints round-trip
through disk and a missing checkpoint returns None (not an exception).
"""

from pathlib import Path

import pytest

from agent_harness.persistence.checkpoint import load_checkpoint, save_checkpoint
from agent_harness.persistence.retry import FatalError, with_backoff


def test_with_backoff_retries_then_succeeds() -> None:
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = with_backoff(flaky, max_retries=5, base_delay=0.01, max_delay=0.02)

    assert result == "ok"
    assert calls["count"] == 3


def test_with_backoff_raises_after_exhausting_retries() -> None:
    calls = {"count": 0}

    def always_fails() -> str:
        calls["count"] += 1
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        with_backoff(always_fails, max_retries=3, base_delay=0.01, max_delay=0.02)

    assert calls["count"] == 3


def test_with_backoff_never_retries_fatal_error() -> None:
    calls = {"count": 0}

    def bad_key() -> str:
        calls["count"] += 1
        raise FatalError("invalid API key")

    with pytest.raises(FatalError):
        with_backoff(bad_key, max_retries=5, base_delay=0.01)

    assert calls["count"] == 1


def test_checkpoint_round_trips_through_disk(tmp_path: Path) -> None:
    save_checkpoint("run-1", {"phase": "EXECUTE", "step_count": 3}, directory=tmp_path)

    loaded = load_checkpoint("run-1", directory=tmp_path)

    assert loaded == {"phase": "EXECUTE", "step_count": 3}


def test_load_missing_checkpoint_returns_none(tmp_path: Path) -> None:
    assert load_checkpoint("does-not-exist", directory=tmp_path) is None
