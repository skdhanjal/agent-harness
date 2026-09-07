"""Smoke test: confirms the package imports and the dev environment is wired up.

Once Module 1 lands, this file can stay as a canary — if it ever breaks,
the environment itself is broken, not the harness logic.
"""

import agent_harness


def test_package_imports() -> None:
    assert agent_harness is not None
