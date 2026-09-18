"""Generic async side-effect runner for app controllers.

Every app has the same shape of side effect: the controller emits a request
record carrying a ``request_id``, something slow runs off the controller (an
engine, a vision model, an LLM), and the reply is submitted back as an event the
controller is already waiting for.

Before this module each app hand-rolled that loop (``spell`` has two copies).
Usage::

    effects = EffectRunner(game.submit, name="chess_effects")
    effects.register(RequestAnalysis, run_analysis)   # async def run_analysis(request) -> event
    game.outputs.to(effects.sink(), name="chess_effects", subs=subs)
    ...
    await effects.close()

``run_analysis`` may return ``None`` to submit nothing. Exceptions are turned
into an event by the optional ``on_error`` handler, or logged and dropped.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mulive.core.logging_utils import monitor_log

EffectHandler = Callable[[Any], Awaitable[Any]]
ErrorHandler = Callable[[Any, Exception], Any]


class EffectRunner:
    """Run registered async handlers for the ``effect`` field of app outputs.

    Args:
        submit: Controller entry point for events, e.g. ``AsyncControllerFlow.submit``.
        name: Label used in log lines.
    """

    def __init__(self, submit: Callable[[Any], Any], name: str = "effects"):
        self.name = name
        self._submit = submit
        self._handlers: dict[type, EffectHandler] = {}
        self._error_handlers: dict[type, ErrorHandler] = {}
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def register(
        self,
        request_type: type,
        handler: EffectHandler,
        on_error: ErrorHandler | None = None,
    ) -> None:
        """Bind one request type to the coroutine that fulfils it.

        Args:
            request_type: Class of the request record the controller emits.
            handler: ``async def handler(request) -> event | None``.
            on_error: ``on_error(request, exc) -> event | None`` used when the
                handler raises. Without it, failures are logged and dropped.

        Raises:
            ValueError: When ``request_type`` is already registered.
        """
        if request_type in self._handlers:
            raise ValueError(f"{self.name}: duplicate effect handler for {request_type.__name__}")
        self._handlers[request_type] = handler
        if on_error is not None:
            self._error_handlers[request_type] = on_error

    def sink(self):
        """Return a stream sink that starts effects for outputs carrying one.

        Returns:
            A callable suitable for ``Stream.to(...)``.
        """

        def attach(observable):
            return observable.subscribe(
                on_next=self.handle_output,
                on_error=lambda error: monitor_log(f"{self.name} stream error: {error}"),
            )

        return attach

    def handle_output(self, output: Any) -> asyncio.Task | None:
        """Start the effect for one app output, if it carries a known request.

        Args:
            output: An ``AppOutput``-shaped object with an ``effect`` attribute.

        Returns:
            The spawned task, or None when there is nothing to run.
        """
        return self.start(getattr(output, "effect", None))

    def start(self, request: Any) -> asyncio.Task | None:
        """Start the effect for one request record.

        Args:
            request: A registered request record, or None.

        Returns:
            The spawned task, or None when the request is None, unregistered, or
            the runner is closed.
        """
        if request is None or self._closed:
            return None
        handler = self._handlers.get(type(request))
        if handler is None:
            monitor_log(f"{self.name} ignoring unregistered effect {type(request).__name__}")
            return None
        task = asyncio.get_running_loop().create_task(self._run(request, handler))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, request: Any, handler: EffectHandler) -> None:
        try:
            event = await handler(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - effect boundary must not kill the session
            monitor_log(
                f"{self.name} effect failed request={type(request).__name__} "
                f"error={type(exc).__name__}: {exc}"
            )
            on_error = self._error_handlers.get(type(request))
            event = on_error(request, exc) if on_error is not None else None
        if event is not None and not self._closed:
            self._submit(event)

    async def close(self) -> None:
        """Cancel every outstanding effect and wait for the tasks to finish."""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
