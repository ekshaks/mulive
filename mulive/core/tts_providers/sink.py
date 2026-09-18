import asyncio
import threading
import time
from typing import Any, Optional, Protocol

from reactivex.disposable import CompositeDisposable, Disposable

from ..logging_utils import monitor_log, monitor_time


class TTSProvider(Protocol):
    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        ...


class PlaybackState:
    """Thread-safe state exposed by TTS while playback is active."""

    def __init__(self):
        self._playing = threading.Event()
        self._last_stopped_at = 0.0

    def set_playing(self, playing: bool):
        if playing:
            self._playing.set()
        else:
            self._last_stopped_at = time.monotonic()
            self._playing.clear()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def is_playing_or_recent(self, cooldown_seconds: float) -> bool:
        return self.is_playing() or (time.monotonic() - self._last_stopped_at) < cooldown_seconds


def _as_observable(stream_or_observable: Any):
    return getattr(stream_or_observable, "observable", stream_or_observable)


def _clear_queue(queue: asyncio.Queue):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def tts_sink(
    provider: TTSProvider,
    interrupts: Optional[Any] = None,
    name: str = "tts",
    clear_queue_on_interrupt: bool = True,
    state: Optional[PlaybackState] = None,
):
    """Reusable sink: queue text, handle interruption, call provider.speak()."""

    def attach(text_observable):
        queue = asyncio.Queue()
        interrupt_event = [asyncio.Event()]
        disposable = CompositeDisposable()

        def on_signal(event):
            print(f"[interrupt] {name} received {event}")
            interrupt_event[0].set()
            if clear_queue_on_interrupt:
                _clear_queue(queue)
            clear_output = getattr(provider, "clear_output", None)
            if clear_output is not None:
                clear_output()
            print(f"[interrupt] {name} interrupt_event set={interrupt_event[0].is_set()}")

        async def worker():
            while True:
                text = await queue.get()
                if text is None:
                    break
                if text and text.strip():
                    try:
                        if interrupt_event[0].is_set():
                            # Each assistant response owns a fresh cancellation
                            # event; no synthetic "speech ended" event is needed.
                            interrupt_event[0] = asyncio.Event()
                        if state is not None:
                            state.set_playing(True)
                        monitor_log(f"[interrupt] {name} event=speak_start")
                        started_at = time.perf_counter()
                        try:
                            await provider.speak(text, interrupt_event[0])
                        finally:
                            monitor_time(
                                "tts",
                                "speak_complete",
                                time.perf_counter() - started_at,
                                provider=type(provider).__name__,
                                interrupted=interrupt_event[0].is_set(),
                            )
                        monitor_log(
                            f"[interrupt] {name} event=speak_end "
                                f"interrupted={interrupt_event[0].is_set()}"
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        print(f"{name} provider error: {exc}")
                    finally:
                        if state is not None:
                            state.set_playing(False)

        task = asyncio.create_task(worker(), name=name)

        if interrupts is not None:
            interrupt_observable = _as_observable(interrupts)
            disposable.add(interrupt_observable.subscribe(on_signal))

        disposable.add(
            text_observable.subscribe(
                on_next=lambda text: queue.put_nowait(text),
                on_error=lambda error: print(f"{name} text stream error: {error}"),
                on_completed=lambda: queue.put_nowait(None),
            )
        )
        disposable.add(Disposable(lambda: task.cancel()))
        return disposable

    return attach
