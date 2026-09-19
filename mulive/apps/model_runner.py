"""Model runners used by application conversation harnesses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
import inspect
import time
from typing import Any, Protocol

from mulive.core.logging_utils import monitor_time


class ModelRunner(Protocol):
    """Run one self-contained model request and return its final text."""

    async def run(self, prompt: str, *, usage_limits: Any = None) -> str:
        """Run a request without retaining conversation history."""


class ModelRunnerUnavailableError(RuntimeError):
    """Indicate that the selected model runner's optional package is absent."""


class PydanticAIModelRunner:
    """Run a Pydantic AI model and its registered function tools."""

    def __init__(
        self,
        model: Any,
        *,
        instructions: str,
        tools: Sequence[Callable[..., Any]] = (),
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        try:
            from pydantic_ai import Agent as ModelAgent
        except ModuleNotFoundError as exc:
            if exc.name != "pydantic_ai":
                raise
            raise ModelRunnerUnavailableError(
                'Install LLM support with: pip install "mulive[groq]"'
            ) from exc

        self._agent = ModelAgent(
            model,
            tools=tuple(_monitored_tool(tool) for tool in tools),
            instructions=instructions,
            output_type=str,
            retries=1,
            model_settings=(
                {"parallel_tool_calls": False}
                if model_settings is None else model_settings
            ),
        )

    async def run(self, prompt: str, *, usage_limits: Any = None) -> str:
        """Run the model with supplied limits or a five-request default."""
        from pydantic_ai.usage import UsageLimits

        try:
            result = await self._agent.run(
                prompt,
                usage_limits=(
                    UsageLimits(request_limit=5, tool_calls_limit=5)
                    if usage_limits is None else usage_limits
                ),
            )
        except ModuleNotFoundError as exc:
            if exc.name not in {"pydantic_ai", "groq"}:
                raise
            raise ModelRunnerUnavailableError(
                'Install LLM support with: pip install "mulive[groq]"'
            ) from exc
        return result.output


def _monitored_tool(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Return a tool that records the time spent in each invocation."""

    @wraps(tool)
    async def run_tool(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        try:
            result = tool(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except BaseException:
            monitor_time(
                "model_tool",
                tool.__name__,
                time.perf_counter() - started_at,
                outcome="failed",
            )
            raise

        monitor_time(
            "model_tool",
            tool.__name__,
            time.perf_counter() - started_at,
            outcome="ok",
        )
        return result

    return run_tool


def create_model_runner(
    model: Any,
    *,
    instructions: str,
    tools: Sequence[Callable[..., Any]] = (),
    model_settings: dict[str, Any] | None = None,
) -> ModelRunner:
    """Build the default model runner for one Mulive conversation."""
    return PydanticAIModelRunner(
        model,
        instructions=instructions,
        tools=tools,
        model_settings=model_settings,
    )
