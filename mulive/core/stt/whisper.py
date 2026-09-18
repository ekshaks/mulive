"""Local Whisper STT providers and their Stream DSL adapter."""

import asyncio
import importlib.util
import time

import numpy as np

from ..events import TranscriptEvent
from ..logging_utils import monitor_log, monitor_time

#: The pip package each local backend needs. ``mlx-whisper`` is Apple-Silicon only
#: and is not in requirements.txt, so a config asking for it on another machine —
#: or on a Python version it has no wheels for — must say so instead of going deaf.
BACKEND_PACKAGES = {"mlx": "mlx_whisper", "faster_whisper": "faster_whisper"}


# Cache Whisper models keyed by (mode, model_size, model_id, frozen kwargs).
# Loading a faster-whisper base int8 model takes ~1-3 s on a 1-vCPU box and
# was previously repeated on every new browser connect because the pipeline
# built a fresh WhisperSTT per session — the visible "hang on URL connect".
_WHISPER_MODEL_CACHE: dict = {}


def _cache_key(mode, model_size, model_id, kwargs):
    return (mode, model_size, model_id, tuple(sorted(kwargs.items())))


def get_faster_whisper_model(model_name: str = "base", compute_type: str = "int8", **kwargs):
    """Load a faster-whisper CTranslate2 model.

    Extra keyword arguments are forwarded to ``faster_whisper.WhisperModel`` so
    callers can tune the CPU-bound settings that matter on small servers
    (``cpu_threads``, ``num_workers``, ``device``, ``device_index``, ...).
    """
    from faster_whisper import WhisperModel

    print(f"Loading faster Whisper model... model={model_name} compute_type={compute_type} kwargs={kwargs}")
    return WhisperModel(model_name, compute_type=compute_type, **kwargs)


def get_whisper_model(mode="faster_whisper", model_size: str = "base", model_id=None, **kwargs):
    """Return a process-wide singleton Whisper model for the given config.

    Second and later callers with the same (mode, model_size, model_id, kwargs)
    share the same underlying model handle, so multiple WebRTC sessions do
    not each reload the weights.
    """
    key = _cache_key(mode, model_size, model_id, kwargs)
    cached = _WHISPER_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if mode == "faster_whisper":
        model = get_faster_whisper_model(model_size, **kwargs)
    elif mode == "mlx":
        from .mlx import get_mlx_whisper_model

        model = get_mlx_whisper_model(model_size=model_size, model_id=model_id)
    else:
        raise ValueError(f"Unknown Whisper mode: {mode}")
    _WHISPER_MODEL_CACHE[key] = model
    return model


def warm_up(mode: str = "faster_whisper", model_size: str = "base", **kwargs) -> None:
    """Preload the Whisper model so the first session doesn't pay for it."""
    started = time.perf_counter()
    get_whisper_model(mode, model_size, **kwargs)
    print(f"Whisper ({mode}/{model_size}) warmed up in {time.perf_counter() - started:.2f} s")


def infer_faster_whisper(audio_data, model, language="en"):
    segments, _ = model.transcribe(audio_data, language=language)
    return " ".join(segment.text for segment in segments)


def infer_whisper(mode, audio_data, model, language="en"):
    if mode == "faster_whisper":
        return infer_faster_whisper(audio_data, model, language=language)
    if mode == "mlx":
        from .mlx import infer_mlx

        return infer_mlx(audio_data, model)
    raise ValueError(f"Unknown Whisper mode: {mode}")


class WhisperSTT:
    """Run a local Whisper model against one completed audio segment."""

    def __init__(self, mode="faster_whisper", model_size: str = "base", language: str = "en", **kwargs):
        self.mode = mode
        self.model_size = model_size
        self.kwargs = kwargs
        self.language = language
        self._model = None
        _ = self.model

    @property
    def model(self):
        if self._model is None:
            self._model = get_whisper_model(self.mode, self.model_size, **self.kwargs)
        return self._model

    def transcribe_turn(self, samples: np.ndarray) -> str:
        if len(samples) == 0:
            return ""
        start_time = time.perf_counter()
        audio_fp32 = samples.astype(np.float32) / 32768.0
        result = infer_whisper(self.mode, audio_fp32, self.model, language=self.language)
        monitor_time(
            "stt",
            "transcribe",
            time.perf_counter() - start_time,
            provider=self.mode,
            model=self.model_size,
        )
        return result


def require_backend(mode: str) -> None:
    """Check that the chosen Whisper backend can actually be imported.

    Without this the missing import surfaces one utterance at a time, inside a
    worker thread, where it is turned into an empty transcript — so the app looks
    like it simply cannot hear, with nothing on screen to explain why.

    Args:
        mode: ``mlx`` or ``faster_whisper``.

    Raises:
        ValueError: When the mode is not a known backend.
        RuntimeError: When the backend's package is not installed.
    """
    package = BACKEND_PACKAGES.get(mode)
    if package is None:
        raise ValueError(f"Unknown Whisper mode: {mode}")
    if importlib.util.find_spec(package) is None:
        alternative = "faster_whisper" if mode == "mlx" else "mlx"
        raise RuntimeError(
            f"STT provider '{mode}' needs the {package.replace('_', '-')} package, "
            f"which is not installed. Install it, or set stt.provider to "
            f"'{alternative}' in the app config."
        )


def notify(on_status, result: str, data: dict) -> None:
    """Report a status to the browser without letting it break the pipeline.

    The callback ends up writing to a WebRTC data channel, which can close at any
    moment. An exception here would error the transcript stream — and that stream is
    never resubscribed, so a closing channel could leave a live session deaf.

    Args:
        on_status: The ``(result, data)`` callback, or None.
        result: ``loading`` / ``ready`` / ``error``.
        data: Extra fields for the browser.
    """
    if on_status is None:
        return
    try:
        on_status(result, data)
    except Exception as exc:  # noqa: BLE001 - a status message is never worth a crash
        monitor_log(f"stt status callback failed result={result} error={type(exc).__name__}: {exc}")


def transcription_failed(reason: str, model_size: str, on_status) -> TranscriptEvent:
    """Report one failed transcription and keep the stream alive.

    Returning an empty final transcript rather than raising matters: the transcript
    stream feeds the controller's only speech input, and a stream that errors is
    never resubscribed — one bad segment would leave the session deaf for good. The
    failure is logged and pushed to the browser instead of vanishing.

    Args:
        reason: What went wrong, in words.
        model_size: The model that was being used, for the log line.
        on_status: Optional ``(result, data)`` callback into the browser.

    Returns:
        An empty final :class:`~mulive.core.events.TranscriptEvent`.
    """
    monitor_log(f"stt transcription failed model={model_size} reason={reason}")
    notify(on_status, "error", {"model_size": model_size, "reason": reason})
    return TranscriptEvent(text="", is_final=True)


def whisper_stt(
    name: str = "whisper_stt",
    model_size: str = "tiny",
    mode: str = "mlx",
    debug_audio_dir=None,
    timeout_s: float = 20.0,
    on_status=None,
    **kwargs,
):
    """Create a final-transcript stage backed by a local Whisper provider.

    Args:
        name: Stage name, for logs.
        model_size: Whisper model size or id.
        mode: ``mlx`` or ``faster_whisper``.
        debug_audio_dir: Optional directory to dump each segment into.
        timeout_s: How long one segment may take before it is given up on.
        on_status: Optional ``(result, data)`` callback: ``loading`` / ``ready`` /
            ``error``, forwarded to the browser.
        **kwargs: Backend options (``compute_type``, ``cpu_threads``, ``language``).

    Returns:
        A Stream DSL stage mapping audio segments to final transcripts.

    Raises:
        RuntimeError: When the backend package is not installed.
        ValueError: When ``mode`` is not a known backend.
    """
    require_backend(mode)
    from ..stream_dsl import _dump_stt_audio, async_map_stage
    from .pinned import PinnedWhisper

    # Both backends run on one pinned worker thread: model loading and inference stay
    # off the event loop, and only one segment is ever being transcribed at a time.
    backend = PinnedWhisper(mode=mode, model_size=model_size, **kwargs)

    async def transcribe_turn(segment):
        context = getattr(segment, "context", None)
        samples = getattr(segment, "samples", segment)
        if debug_audio_dir:
            _dump_stt_audio(samples, debug_audio_dir)
        if backend.is_loading():
            notify(on_status, "loading", {"model_size": model_size})
        try:
            text = await asyncio.wait_for(backend.transcribe_turn(samples), timeout=timeout_s)
        except asyncio.TimeoutError:
            return transcription_failed(f"timed out after {timeout_s:g}s", model_size, on_status)
        except Exception as exc:  # noqa: BLE001 - one bad segment must not deafen us
            return transcription_failed(f"{type(exc).__name__}: {exc}", model_size, on_status)
        notify(on_status, "ready", {"model_size": model_size})
        return TranscriptEvent(text=text or "", is_final=True, context=context)

    return async_map_stage(transcribe_turn, name=name, on_dispose=backend.shutdown)
