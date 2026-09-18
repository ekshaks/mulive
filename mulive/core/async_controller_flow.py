import asyncio
from collections import deque
from threading import Lock
from typing import Any, Callable, Optional, Protocol

from reactivex.subject import Subject

from .stream_dsl import Stream


EventPredicate = Callable[[Any], bool]


class AsyncController(Protocol):
    """Contract for coroutine-style controllers."""

    async def run(self, ctx: "AsyncControllerContext") -> None:
        ...


class AsyncControllerContext:
    """Narrow API exposed to async workflow controllers."""

    def __init__(self, flow: "AsyncControllerFlow"):
        self._flow = flow

    async def wait_for(self, predicate: type | tuple[type, ...] | EventPredicate) -> Any:
        """Wait until the next queued event matching ``predicate`` arrives."""
        return await self._flow._wait_for(predicate)

    async def next_event(self) -> Any:
        """Return the next queued event without filtering."""
        return await self._flow._next_event()

    def emit(self, output: Any) -> None:
        """Publish one output packet."""
        self._flow._emit(output)


class AsyncControllerFlow:
    """Run one async workflow controller behind the same stream-facing shape as ControllerFlow.

    Usage:
        flow = AsyncControllerFlow(MyWorkflow(), name="my_workflow")
        voice_events.to(flow.input_sink(), subs=subs)
        button_events.to(flow.input_sink(), subs=subs)
        flow.start()

    The workflow implements ``async run(ctx)`` and emits output packets with
    ``ctx.emit(...)``. Events are serialized through an internal asyncio queue
    and consumed with ``await ctx.next_event()`` or ``await ctx.wait_for(...)``.
    """

    def __init__(self, controller: AsyncController, name: str = "async_controller"):
        self.controller = controller
        self.name = name
        self._output_subject = Subject()
        self._has_input = False
        self._started = False
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None
        self._pending_events = deque()
        self._lock = Lock()
        self.outputs = Stream.source(
            self._output_subject,
            name=f"{name}.outputs",
        )

    def input_sink(self):
        """Return a sink that forwards one event stream into the workflow queue."""

        def attach(observable):
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            self._has_input = True

            return observable.subscribe(
                on_next=self.submit,
                on_error=lambda error: print(
                    f"{self.name} input stream error: {error}"
                ),
            )

        return attach

    def submit(self, item: Any) -> bool:
        """Queue one event for the async workflow.

        Returns ``False`` when the flow is closed.
        """

        with self._lock:
            if self._closed:
                return False
            if not self._started or self._loop is None or self._event_queue is None:
                self._pending_events.append(item)
                return True
            queue = self._event_queue

        # All producers (stream sinks, effect callbacks) run on the event
        # loop, so the queue can be fed directly.
        queue.put_nowait(item)
        return True

    def start(self) -> None:
        if not self._has_input:
            raise RuntimeError(f"{self.name} must have an input before start")
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            if self._started:
                return
            self._loop = asyncio.get_running_loop()
            self._event_queue = asyncio.Queue()
            pending = list(self._pending_events)
            self._pending_events.clear()
            self._started = True

        for item in pending:
            self._event_queue.put_nowait(item)

        ctx = AsyncControllerContext(self)
        self._task = self._loop.create_task(self._run_controller(ctx))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending_events.clear()
            task = self._task
            queue = self._event_queue

        if task is not None and not task.done():
            task.cancel()
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
        self._output_subject.on_completed()

    async def _run_controller(self, ctx: AsyncControllerContext) -> None:
        try:
            await self.controller.run(ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with self._lock:
                self._closed = True
                self._pending_events.clear()
            self._output_subject.on_error(exc)
        else:
            with self._lock:
                self._closed = True
            self._output_subject.on_completed()

    async def _wait_for(self, predicate: type | tuple[type, ...] | EventPredicate) -> Any:
        while True:
            event = await self._next_event()
            if self._matches(predicate, event):
                return event

    async def _next_event(self) -> Any:
        queue = self._event_queue
        if queue is None:
            raise RuntimeError(f"{self.name} is not started")
        return await queue.get()

    def _emit(self, output: Any) -> None:
        if self._closed:
            return
        self._output_subject.on_next(output)

    @staticmethod
    def _matches(predicate: type | tuple[type, ...] | EventPredicate, event: Any) -> bool:
        if isinstance(predicate, type):
            return isinstance(event, predicate)
        if isinstance(predicate, tuple) and all(isinstance(item, type) for item in predicate):
            return isinstance(event, predicate)
        return predicate(event)
