"""Verification tests for Module 2: Tool Calling.

Proves the three claims the module makes:
1. valid args execute normally
2. invalid args / unknown tools / raised exceptions all become structured
   errors, never uncaught exceptions
3. a hanging tool is cut off by the timeout, not waited out
"""

import time

import pytest

from agent_harness.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_valid_args_execute_successfully(registry: ToolRegistry) -> None:
    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = registry.execute("add", {"a": 2, "b": 3})

    assert result.ok is True
    assert result.result == 5


def test_invalid_args_return_structured_error_not_exception(registry: ToolRegistry) -> None:
    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = registry.execute("add", {"a": "not_a_number", "b": 3})

    assert result.ok is False
    assert result.error is not None
    assert "invalid_args" in result.error


def test_unknown_tool_returns_structured_error(registry: ToolRegistry) -> None:
    result = registry.execute("does_not_exist", {})

    assert result.ok is False
    assert result.error is not None
    assert "unknown_tool" in result.error


def test_tool_exception_is_caught_not_raised(registry: ToolRegistry) -> None:
    @registry.tool
    def divide(a: int, b: int) -> float:
        """Divide a by b."""
        return a / b

    result = registry.execute("divide", {"a": 1, "b": 0})

    assert result.ok is False
    assert result.error is not None
    assert "execution_error" in result.error


def test_slow_tool_times_out_instead_of_hanging(registry: ToolRegistry) -> None:
    @registry.tool
    def slow_tool() -> str:
        """Sleeps far longer than the timeout."""
        time.sleep(5)
        return "done"

    start = time.monotonic()
    result = registry.execute("slow_tool", {}, timeout_s=1.0)
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.error is not None
    assert "timeout" in result.error
    assert elapsed < 2.0  # proves we didn't wait out the full 5s sleep


def test_schema_for_llm_reflects_registered_tools(registry: ToolRegistry) -> None:
    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    schemas = registry.schema_for_llm()

    assert len(schemas) == 1
    assert schemas[0]["name"] == "add"
    parameters = schemas[0]["parameters"]
    assert isinstance(parameters, dict)
    assert "a" in parameters["properties"]
