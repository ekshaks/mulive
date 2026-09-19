import asyncio
import time
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import reactivex
from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable

from .logging_utils import monitor_log


class Sub:
    """Disposable returned by `.to(...)`."""

    def __init__(self, disposable: Any, name: Optional[str] = None):
        self.disposable = disposable
        self.name = name
        self.disposed = False

    def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        if self.disposable is not None:
            self.disposable.dispose()


class SubGroup:
    """Session-level owner for all active stream subscriptions."""

    def __init__(self):
        self._subs = []
        self._async_closers = []

    def add(self, sub: Sub) -> Sub:
        if sub not in self._subs:
            self._subs.append(sub)
        return sub

    def dispose(self):
        for sub in list(self._subs):
            sub.dispose()
        self._subs.clear()

    def add_async_close(self, closer):
        self._async_closers.append(closer)

    async def aclose(self):
        self.dispose()
        closers = list(reversed(self._async_closers))
        self._async_closers.clear()
        for close in closers:
            await close()


class Stream:
    """Small wrapper that adds `|` and `.to(...)` on top of an RxPY observable."""

    def __init__(self, observable, name: Optional[str] = None):
        self.observable = observable
        self.name = name

    @classmethod
    def source(cls, observable, name: Optional[str] = None):
        return cls(observable, name=name)

    def __or__(self, stage):
        return stage(self)

    def to(self, sink, name: Optional[str] = None, subs: Optional[SubGroup] = None) -> Sub:
        sink_disposable = sink(self.observable)
        disposable = CompositeDisposable()
        if sink_disposable is not None:
            disposable.add(sink_disposable)
        sub = Sub(disposable, name=name or self.name)
        if subs is not None:
            subs.add(sub)
        return sub

    def pipe(self, *pipe_ops, name: Optional[str] = None):
        return Stream(
            self.observable.pipe(*pipe_ops),
            name=name,
        )

    def latest(self, name: Optional[str] = None, subs: Optional[SubGroup] = None):
        # Side input helper. It samples the newest value when another stage asks.
        latest_value = LatestValue(name=name or self.name)
        upstream_sub = self.to(latest_value.sink(), name=f"{name or self.name}.latest", subs=subs)
        latest_value._sub = upstream_sub
        return latest_value


class MultiOutput:
    """Named outputs for stages like VAD that produce data plus side signals."""

    def __init__(self, *, deprecated_aliases=None, **streams: Stream):
        self._streams = streams
        self._deprecated_aliases = deprecated_aliases or {}

    def __getattr__(self, name: str):
        target = self._deprecated_aliases.get(name)
        if target is not None:
            warnings.warn(
                f"turn.{name} is deprecated; use turn.{target} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._streams[target]
        try:
            return self._streams[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class LatestValue:
    value: Any = None
    name: Optional[str] = None
    _sub: Optional[Sub] = None

    def get(self):
        return self.value

    def sink(self):
        def attach(observable):
            return observable.subscribe(lambda item: setattr(self, "value", item))
        return attach

    def dispose(self):
        if self._sub is not None:
            self._sub.dispose()


def turn_detector(name: str = "turn_detector", **kwargs):
    def apply(stream: Stream):
        from .turn import SpeechStarted, VoiceTurn
        from .turndet import turn_detector_vad

        events = turn_detector_vad(stream.observable, **kwargs).pipe(ops.share())

        return MultiOutput(
            audio=Stream(
                events.pipe(
                    ops.filter(lambda event: isinstance(event, VoiceTurn)),
                    ops.map(lambda event: event.samples),
                ),
                name=f"{name}.audio",
            ),
            started=Stream(
                events.pipe(
                    ops.filter(lambda event: isinstance(event, SpeechStarted)),
                ),
                name=f"{name}.started",
            ),
            events=Stream(events, name=f"{name}.events"),
            value=Stream(
                events.pipe(
                    ops.filter(lambda event: isinstance(event, VoiceTurn)),
                ),
                name=f"{name}.value",
            ),
            deprecated_aliases={"segments": "audio", "turns": "value", "signals": "started"},
        )

    return apply


def _dump_stt_audio(segment, directory: str, rate: int = 16000) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"stt-{time.time_ns()}.wav"
    samples = np.asarray(segment)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.reshape(-1).tobytes())
    print(f"[stt-debug] wrote {path} samples={samples.size}")
    return path


def check_stt_provider(provider: str) -> None:
    """Check that an ``stt.provider`` value can run in this installation.

    Call this while an app is loading, next to the other "can this app run here?"
    checks. A provider whose package is missing then reports itself once, in the
    server log and on the dashboard, instead of turning every single thing the user
    says into an empty transcript.

    Args:
        provider: The configured provider name.

    Raises:
        ValueError: When the provider is unknown.
        RuntimeError: When a local Whisper backend's package is not installed.
    """
    provider = (provider or "").lower()
    if provider in {"mlx", "faster_whisper"}:
        from .stt.whisper import require_backend

        require_backend(provider)
        return
    if provider != "deepgram":
        raise ValueError(f"Unknown STT provider: {provider}")


def stt(config, *, name: str = "stt", on_status=None, debug_audio_dir=None):
    """Create an STT stage from one :class:`STTConfig`."""
    from .stt.config import STTConfig

    if not isinstance(config, STTConfig):
        raise TypeError("stt requires an STTConfig")
    if config.provider == "deepgram":
        from .stt.deepgram import deepgram_stt

        return deepgram_stt(config, name=name, on_status=on_status)
    if config.provider in {"mlx", "faster_whisper"}:
        from .stt.whisper import whisper_stt

        return whisper_stt(
            config,
            name=name,
            on_status=on_status,
            debug_audio_dir=debug_audio_dir,
        )
    raise ValueError(f"Unknown STT provider: {config.provider}")


def whisper_stt(*args, **kwargs):
    """Compatibility alias for the local Whisper STT stage."""

    from .stt.whisper import whisper_stt as local_whisper_stt

    return local_whisper_stt(*args, **kwargs)


def drop_while(predicate: Callable[[], bool], name: str = "drop_while"):
    """Drop stream items while a zero-argument predicate is true."""

    def apply(stream: Stream):
        return stream.pipe(ops.filter(lambda item: not predicate()), name=name)

    return apply


def filter_items(predicate: Callable[[Any], bool], name: str = "filter_items"):
    """Keep stream items that satisfy an item-level predicate."""

    def apply(stream: Stream):
        return stream.pipe(ops.filter(predicate), name=name)

    return apply


def map_items(mapper: Callable[[Any], Any], name: str = "map_items"):
    """Map each stream item with a synchronous item-level function."""

    def apply(stream: Stream):
        return stream.pipe(ops.map(mapper), name=name)

    return apply


def map_filter_items(
    map_fn: Callable[[Any], Any],
    filter_fn: Callable[[Any], bool],
    name: str = "map_filter_items",
):
    """Map each item, then keep mapped values that satisfy ``filter_fn``."""

    def apply(stream: Stream):
        return stream.pipe(
            ops.map(map_fn),
            ops.filter(filter_fn),
            name=name,
        )

    return apply


def expand_items(
    expand_fn: Callable[[Any], Any],
    name: str = "expand_items",
):
    """Expand each stream item into a synchronous sequence of output items."""

    def apply(stream: Stream):
        return stream.pipe(
            ops.flat_map(lambda item: reactivex.from_iterable(expand_fn(item))),
            name=name,
        )

    return apply


def final_transcript_text(name: str = "final_transcript_text"):
    """Select text from the canonical final-transcript filter."""

    def apply(stream: Stream):
        return final_transcripts()(stream).pipe(
            ops.map(lambda event: event.text),
            name=name,
        )

    return apply


def final_transcripts(name: str = "final_transcripts"):
    """Select final non-empty transcript events while retaining their context."""

    def apply(stream: Stream):
        return stream.pipe(
            ops.filter(lambda event: event.is_final and bool(event.text and event.text.strip())),
            name=name,
        )

    return apply


def async_map_stage(func, name: str = "async_stage", concurrency: str = "serial", on_dispose: Optional[Callable[[], None]] = None):
    """Create an RxPY stage that awaits one coroutine call per input item.

    A normal Rx ``map`` is synchronous: mapping an item with an ``async``
    function would emit a coroutine object rather than await it. This adapter
    converts each coroutine call into an inner Observable with
    ``defer``/``from_future`` and lets RxPY control how those inner operations
    are flattened:

    - ``serial`` processes inputs in order, one at a time.
    - ``parallel`` allows operations to overlap.
    - ``latest`` cancels stale work when a newer item arrives.
    - ``drop`` ignores new items while one operation is active.

    The current asyncio loop is captured when the stage is built. All pipeline
    callbacks run on that loop (sources and the VAD timer are loop-scheduled),
    so each coroutine becomes a plain task on it. The output is shared so
    multiple sinks reuse one async execution rather than invoking ``func``
    once per sink.
    Disposing the final subscription cancels active futures; ``on_dispose``
    optionally releases provider resources such as a model executor.
    """

    supported = {"serial", "parallel", "latest", "drop"}
    if concurrency not in supported:
        raise ValueError(f"Unknown concurrency policy: {concurrency}")

    def apply(stream: Stream):
        loop = asyncio.get_running_loop()

        def async_observable(item):
            def future_factory(_scheduler=None):
                return reactivex.from_future(loop.create_task(func(item)))

            return reactivex.defer(future_factory)

        inner_streams = stream.observable.pipe(ops.map(async_observable))
        if concurrency == "serial":
            output = inner_streams.pipe(ops.merge(max_concurrent=1))
        elif concurrency == "parallel":
            output = inner_streams.pipe(ops.flat_map())
        elif concurrency == "latest":
            output = inner_streams.pipe(ops.switch_latest())
        else:
            output = inner_streams.pipe(ops.exclusive())

        if on_dispose is not None:
            output = output.pipe(ops.finally_action(on_dispose))

        return Stream(output.pipe(ops.share()), name=name)

    return apply


def print_sink(prefix: str = ""):
    def attach(observable):
        return observable.subscribe(lambda item: print(f"{prefix}{item}"))
    return attach


def client_message_sink(session, channel: str = "server_text"):
    """Sink that forwards each stream item to the browser data channel."""

    def attach(observable):
        def on_next(message):
            monitor_log(f"sending {type(message).__name__} to client")
            try:
                session.send_to_client(message, channel=channel)
            except Exception as exc:
                print(f"Error sending client message: {exc}")

        def on_error(error):
            print(f"client_message_sink error: {error}")

        return observable.subscribe(on_next=on_next, on_error=on_error)

    return attach
