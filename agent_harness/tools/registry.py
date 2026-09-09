"""Tool Calling: give the agent the ability to act, safely.

The harness owns three things the raw LLM API doesn't:
1. Auto-deriving a JSON schema from a plain Python function's type hints
   -- no hand-written schema to keep in sync with the real signature.
2. Validating LLM-supplied arguments against that schema *before* the
   function runs.
3. Executing inside a sandbox: exceptions are caught and a timeout is
   enforced, so one bad or hanging tool call can't take down the loop.
"""

from __future__ import annotations

import concurrent.futures
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


class ToolExecutionResult(BaseModel):
    """The only shape a tool call ever returns -- success or a structured error."""

    ok: bool
    result: object = None
    error: str | None = None


class ToolRegistry:
    """Maps tool names to (callable, auto-derived pydantic arg model)."""

    def __init__(self, max_workers: int = 4) -> None:
        self._tools: dict[str, tuple[Callable[..., object], type[BaseModel]]] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def tool(self, fn: Callable[..., object]) -> Callable[..., object]:
        """Decorator: registers `fn` as a callable tool for the agent."""
        arg_model = self._build_arg_model(fn)
        self._tools[fn.__name__] = (fn, arg_model)
        return fn

    def _build_arg_model(self, fn: Callable[..., object]) -> type[BaseModel]:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            annotation = hints.get(name, Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (annotation, default)
        model: type[BaseModel] = create_model(f"{fn.__name__}_Args", **fields)
        return model

    def schema_for_llm(self) -> list[dict[str, object]]:
        """Tool schemas in a provider-agnostic shape; adapt per-provider at the edge."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": inspect.getdoc(fn) or "",
                    "parameters": model.model_json_schema(),
                },
            }
            for name, (fn, model) in self._tools.items()
        ]

    def execute(
        self, name: str, raw_args: dict[str, object], timeout_s: float = 10.0
    ) -> ToolExecutionResult:
        """Validate args, run the tool, and never let it raise or hang."""
        if name not in self._tools:
            return ToolExecutionResult(ok=False, error=f"unknown_tool: '{name}' is not registered")

        fn, arg_model = self._tools[name]

        try:
            validated = arg_model.model_validate(raw_args)
        except ValidationError as e:
            return ToolExecutionResult(ok=False, error=f"invalid_args: {e}")

        future = self._executor.submit(fn, **validated.model_dump())
        try:
            result = future.result(timeout=timeout_s)
            return ToolExecutionResult(ok=True, result=result)
        except concurrent.futures.TimeoutError:
            return ToolExecutionResult(
                ok=False, error=f"timeout: tool '{name}' exceeded {timeout_s}s"
            )
        except Exception as e:
            return ToolExecutionResult(ok=False, error=f"execution_error: {e}")
