from collections import deque
from threading import Lock
from typing import Any, Protocol

from reactivex.subject import Subject

from .stream_dsl import Stream


class Controller(Protocol):
    """Contract for a stateful, application-initiated controller."""

    def start(self) -> Any:
        ...

    def handle_input(self, item: Any) -> Any:
        ...


class ControllerFlow:
    """Connect event streams to one serialized controller and shared output.

    Usage:
        flow = ControllerFlow(MyController(), name="my_controller")
        voice_events.to(flow.input_sink(), subs=subs)
        button_events.to(flow.input_sink(), subs=subs)
        flow.start() # start conversation without user input

    Async callbacks can feed the same controller mailbox with ``submit()``.
    Input completion and errors do not close the flow; its owner calls
    ``close()`` when the application or session finishes.
    """

    def __init__(self, controller: Controller, name: str = "controller"):
        self.controller = controller
        self.name = name
        self._output_subject = Subject()
        self._has_input = False
        self._started = False
        self._closed = False
        self._input_queue = deque()
        self._input_lock = Lock()
        self._draining = False
        self.outputs = Stream.source(
            self._output_subject,
            name=f"{name}.outputs",
        )

    def input_sink(self):
        """Return a sink that forwards one event stream into the mailbox."""

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
        """Queue one event for serial controller processing.

        Returns ``False`` when the flow is already closed.
        """

        with self._input_lock:
            if self._closed:
                return False
            self._input_queue.append(item)
            if self._draining:
                return True
            self._draining = True

        while True:
            with self._input_lock:
                if self._closed or not self._input_queue:
                    self._draining = False
                    return not self._closed
                next_item = self._input_queue.popleft()

            try:
                output = self.controller.handle_input(next_item)
                self._output_subject.on_next(output)
            except Exception as exc:
                with self._input_lock:
                    self._closed = True
                    self._input_queue.clear()
                    self._draining = False
                self._output_subject.on_error(exc)
                return False

    def start(self) -> None:
        if not self._has_input:
            raise RuntimeError(f"{self.name} must have an input before start")
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
        if self._started:
            return
        self._started = True
        try:
            self._output_subject.on_next(self.controller.start())
        except Exception as exc:
            self._output_subject.on_error(exc)

    def close(self) -> None:
        with self._input_lock:
            if self._closed:
                return
            self._closed = True
            self._input_queue.clear()
        self._output_subject.on_completed()
