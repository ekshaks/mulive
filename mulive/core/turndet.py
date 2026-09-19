import asyncio
import os
import time
import uuid

import numpy as np
import reactivex
from reactivex.disposable import CompositeDisposable, Disposable
from reactivex.scheduler.eventloop import AsyncIOScheduler
from typing import Any, Callable, Optional, Tuple

from .turn import SpeechStarted, TurnEvent, VoiceTurn

# Singleton instances
_VAD_MODEL = None
_VAD_UTILS = None


def _load_silero_torch():
    """Load Silero VAD via torch.hub. Same shape it has always returned."""
    import torch  # imported lazily so the onnx backend can skip the torch dep

    return torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)


# Pinned upstream copy of silero_vad.onnx. The URL is stable (a git tag on
# snakers4/silero-vad) and the sha256 lets us fail loudly if the bytes ever
# change under us. Bump both together on upgrade.
_SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/"
    "src/silero_vad/data/silero_vad.onnx"
)
_SILERO_ONNX_SHA256 = (
    "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
)
_SILERO_ONNX_BYTES = 2327524


def _mulive_cache_dir():
    """Return (and create) the directory used to cache downloaded VAD assets."""
    from pathlib import Path

    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache = Path(base) / "mulive"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_silero_onnx(dest):
    """Download the pinned Silero VAD ONNX to *dest* and verify its sha256.

    Writes to ``dest.with_suffix('.part')`` first, verifies the hash, then
    atomically renames — so a partial download from a Ctrl-C never presents
    as a valid cached model.
    """
    import hashlib
    import urllib.request
    from pathlib import Path

    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(
        f"Silero VAD ONNX not found locally; downloading {_SILERO_ONNX_BYTES / 1024:.0f} KB "
        f"from {_SILERO_ONNX_URL} -> {dest}"
    )
    with urllib.request.urlopen(_SILERO_ONNX_URL, timeout=30) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _SILERO_ONNX_SHA256:
        raise RuntimeError(
            f"Downloaded Silero VAD ONNX has sha256 {digest}, expected "
            f"{_SILERO_ONNX_SHA256}. Refusing to cache; set "
            f"SILERO_ONNX_MODEL_PATH to a trusted file instead."
        )
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _find_silero_onnx_model():
    """Locate the Silero VAD ONNX model file without importing torch.

    Resolution order:

    1. ``SILERO_ONNX_MODEL_PATH`` env var (must exist).
    2. The ``silero-vad`` PyPI wheel's bundled ``data/silero_vad.onnx``, if
       installed (found via ``find_spec`` so we do not execute its
       torch-importing ``__init__``).
    3. A cached copy at ``$XDG_CACHE_HOME/mulive/silero_vad.onnx``.
    4. Fresh download of the pinned upstream copy into the cache dir, with
       sha256 verification.

    So on a fresh AWS box the ONNX backend needs only ``pip install
    onnxruntime`` — the model is fetched on first use (~2.3 MB). Air-gapped
    deployments can still point ``SILERO_ONNX_MODEL_PATH`` at a preloaded
    file to skip the network round-trip.
    """
    import importlib.util
    from pathlib import Path

    env_path = os.environ.get("SILERO_ONNX_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"SILERO_ONNX_MODEL_PATH does not exist: {path}")
        return path
    spec = importlib.util.find_spec("silero_vad")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = Path(location) / "data" / "silero_vad.onnx"
            if candidate.exists():
                return candidate
    cached = _mulive_cache_dir() / "silero_vad.onnx"
    if cached.exists():
        return cached
    return _download_silero_onnx(cached)


class _NumpyOnnxVad:
    """Silero VAD on onnxruntime with numpy I/O — no torch dependency.

    Mirrors the reference OnnxWrapper: 512-sample windows at 16 kHz (256 at
    8 kHz) with a 64-sample rolling context and a (2, 1, 128) recurrent state.
    """

    def __init__(self, model_path):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )

    def speech_probs(self, audio_fp32, rate):
        """Return the per-window speech probability list for one audio clip."""
        window = 512 if rate == 16000 else 256
        context_size = 64 if rate == 16000 else 32
        audio = np.asarray(audio_fp32, dtype=np.float32).reshape(-1)
        if len(audio) % window:
            audio = np.pad(audio, (0, window - len(audio) % window))
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_size), dtype=np.float32)
        sr = np.array(rate, dtype=np.int64)
        probs = []
        for start in range(0, len(audio), window):
            chunk = audio[start : start + window][None, :]
            x = np.concatenate([context, chunk], axis=1)
            out, state = self.session.run(None, {"input": x, "state": state, "sr": sr})
            context = x[:, -context_size:]
            probs.append(float(out[0, 0]))
        return probs


def _onnx_get_speech_timestamps(
    audio,
    model,
    sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
    **_,
):
    """Numpy segmenter over :class:`_NumpyOnnxVad` probabilities.

    Same call signature as silero's ``get_speech_timestamps`` (as used by
    ``_build_is_speech``); returns a list of ``{"start", "end"}`` sample
    ranges. ``speech_pad_ms`` is accepted for compatibility but padding is
    irrelevant to the boolean is-speech decision this pipeline makes.
    """
    window = 512 if sampling_rate == 16000 else 256
    probs = model.speech_probs(audio, sampling_rate)
    min_speech_frames = max(1, int(min_speech_duration_ms * sampling_rate / 1000 / window))
    min_silence_frames = max(1, int(min_silence_duration_ms * sampling_rate / 1000 / window))
    neg_threshold = max(threshold - 0.15, 0.01)

    segments = []
    current_start = None
    silence_run = 0
    for index, prob in enumerate(probs):
        if prob >= threshold:
            if current_start is None:
                current_start = index
            silence_run = 0
        elif current_start is not None and prob < neg_threshold:
            silence_run += 1
            if silence_run >= min_silence_frames:
                end = index - silence_run + 1
                if end - current_start >= min_speech_frames:
                    segments.append({"start": current_start * window, "end": end * window})
                current_start = None
                silence_run = 0
    if current_start is not None and len(probs) - current_start >= min_speech_frames:
        segments.append({"start": current_start * window, "end": len(probs) * window})
    return segments


def _load_silero_onnx():
    """Load Silero VAD on onnxruntime + numpy, with no torch dependency.

    Returns (model, utils) where utils[0] is a get_speech_timestamps-shaped
    callable so call sites do not need to know which backend they are on.
    """
    model = _NumpyOnnxVad(_find_silero_onnx_model())
    return model, (_onnx_get_speech_timestamps,)


def warm_up_vad() -> None:
    """Load the Silero VAD model and run one dummy inference.

    The ONNX backend can also download the model on first use, which then
    happens the first time the microphone is turned on and looks like a
    hang. Preloading at server start moves the cost off the hot path.
    """
    started = time.perf_counter()
    model, utils = get_vad_model()
    silence = np.zeros(16000, dtype=np.float32)  # 1 s of silence @ 16 kHz
    utils[0](silence, model, sampling_rate=16000)
    print(f"Silero VAD warmed up in {time.perf_counter() - started:.2f} s")


def get_vad_model() -> Tuple[Any, Any]:
    """Return (model, utils) for the Silero VAD backend.

    Backend selection: env ``SILERO_BACKEND`` = ``onnx`` (default) | ``torch``.
    ONNX avoids the Torch dependency and is the portable package default.
    """
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None or _VAD_UTILS is None:
        backend = os.environ.get("SILERO_BACKEND", "onnx").lower()
        print(f"Loading Silero VAD model (backend={backend})...")
        if backend == "onnx":
            _VAD_MODEL, _VAD_UTILS = _load_silero_onnx()
        elif backend == "torch":
            _VAD_MODEL, _VAD_UTILS = _load_silero_torch()
        else:
            raise ValueError(f"Unknown SILERO_BACKEND: {backend!r} (expected 'torch' or 'onnx')")
    return _VAD_MODEL, _VAD_UTILS


class TurnDetector:
    """Detect and collect one user turn from arbitrary PCM16 chunks.

    The detector contains the transport-independent turn rules. Transports
    choose how to schedule :meth:`flush_if_ready` and how to deliver events.
    """

    def __init__(
        self,
        is_speech: Callable[[np.ndarray], bool],
        *,
        silence_timeout: float = 1.0,
        max_samples: int | None = 16_000 * 60,
        analysis_samples: int = 1_600,
    ) -> None:
        self.is_speech = is_speech
        self.silence_timeout = silence_timeout
        self.max_samples = max_samples
        self.analysis_samples = analysis_samples
        self._analysis = np.empty(0, dtype=np.int16)
        self._buffer: list[np.ndarray] = []
        self._turn_id: str | None = None
        self._last_speech_time = 0.0
        self._samples = 0

    @property
    def turn_id(self) -> str | None:
        """Return the active turn identifier, if speech is being buffered."""
        return self._turn_id

    def feed(self, chunk: np.ndarray) -> list[TurnEvent]:
        """Process one PCM16 chunk and return any immediate detector events."""
        samples = np.asarray(chunk, dtype=np.int16).reshape(-1)
        if not samples.size:
            return []
        self._analysis = np.concatenate((self._analysis, samples))
        events: list[TurnEvent] = []
        while self._analysis.size >= self.analysis_samples:
            frame = self._analysis[: self.analysis_samples]
            self._analysis = self._analysis[self.analysis_samples :]
            events.extend(self._process_frame(frame))
        return events

    def flush_if_ready(self, *, force: bool = False) -> list[TurnEvent]:
        """Complete the active turn when its endpoint timeout has elapsed."""
        if not force and (
            self._turn_id is None
            or time.monotonic() - self._last_speech_time < self.silence_timeout
        ):
            return []
        events = self._process_remainder()
        if not force and self._turn_id is not None:
            if time.monotonic() - self._last_speech_time < self.silence_timeout:
                return events
        completed = self._complete()
        if completed is not None:
            events.append(completed)
        return events

    def cancel(self) -> None:
        """Discard buffered and unanalyzed audio."""
        self._analysis = np.empty(0, dtype=np.int16)
        self._buffer.clear()
        self._turn_id = None
        self._samples = 0
        self._last_speech_time = 0.0

    def _process_remainder(self) -> list[TurnEvent]:
        if not self._analysis.size:
            return []
        frame = self._analysis
        self._analysis = np.empty(0, dtype=np.int16)
        padded = np.pad(frame, (0, self.analysis_samples - frame.size))
        if self.is_speech(padded):
            return self._accept_speech(frame)
        return self._accept_silence(frame)

    def _process_frame(self, frame: np.ndarray) -> list[TurnEvent]:
        if self.is_speech(frame):
            return self._accept_speech(frame)
        return self._accept_silence(frame)

    def _accept_speech(self, samples: np.ndarray) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if self._turn_id is None:
            self._turn_id = uuid.uuid4().hex
            events.append(SpeechStarted(self._turn_id))
        if self.max_samples is not None and self._samples + samples.size > self.max_samples:
            completed = self._complete()
            if completed is not None:
                events.append(completed)
            self._turn_id = uuid.uuid4().hex
            events.append(SpeechStarted(self._turn_id))
        self._buffer.append(samples.copy())
        self._samples += samples.size
        self._last_speech_time = time.monotonic()
        return events

    def _accept_silence(self, samples: np.ndarray) -> list[TurnEvent]:
        if self._turn_id is None:
            return []
        if self.max_samples is not None and self._samples + samples.size > self.max_samples:
            completed = self._complete()
            return [completed] if completed is not None else []
        self._buffer.append(samples.copy())
        self._samples += samples.size
        return []

    def _complete(self) -> VoiceTurn | None:
        if self._turn_id is None or not self._buffer:
            return None
        audio = np.concatenate(self._buffer, axis=0)
        turn = VoiceTurn(self._turn_id, audio.astype("<i2", copy=False).tobytes())
        self._buffer.clear()
        self._turn_id = None
        self._samples = 0
        self._last_speech_time = 0.0
        return turn


def _build_is_speech(
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    rate: int,
) -> Callable[[np.ndarray], bool]:
    vad_model, utils = get_vad_model()
    get_speech_timestamps = utils[0]

    def is_speech(samples: np.ndarray) -> bool:
        samples_fp32 = samples.astype(np.float32) / 32768.0
        speech_ts = get_speech_timestamps(
            samples_fp32,
            vad_model,
            sampling_rate=rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        return len(speech_ts) > 0

    return is_speech


def turn_detector_vad(
    audio_observable,
    silence_timeout: float = 1.0,
    poll_interval: float = 0.1,
    min_speech_duration_ms: int = 100, min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 200, threshold: float = 0.4, RATE: int = 16000,
    is_speech_fn: Optional[Callable[[np.ndarray], bool]] = None,
):
    """Return a cold VAD event observable with subscription-owned resources."""

    is_speech = is_speech_fn or _build_is_speech(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        rate=RATE,
    )

    def subscribe(observer, scheduler=None):
        # The silence-poll timer runs on the asyncio event loop, so every
        # callback below (audio on_next, timer tick, dispose) is serialized on
        # one loop and no locking is needed. Audio sources must also emit on
        # this loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "turn_detector_vad must be subscribed from a running asyncio "
                "event loop; its poll timer is scheduled on that loop."
            ) from None

        disposed = [False]
        detector = TurnDetector(
            is_speech,
            silence_timeout=silence_timeout,
        )

        def process_chunk(chunk: np.ndarray):
            if disposed[0]:
                return
            for event in detector.feed(chunk):
                if isinstance(event, SpeechStarted):
                    print("[interrupt] VAD SPEECH_START")
                observer.on_next(event)

        def emit_turn_if_ready(force: bool = False):
            if disposed[0]:
                return
            for event in detector.flush_if_ready(force=force):
                observer.on_next(event)

        def on_completed():
            emit_turn_if_ready(force=True)
            observer.on_completed()

        source_sub = audio_observable.subscribe(
            on_next=process_chunk,
            on_error=observer.on_error,
            on_completed=on_completed,
        )
        timer_sub = reactivex.interval(
            poll_interval, scheduler=AsyncIOScheduler(loop)
        ).subscribe(lambda _: emit_turn_if_ready())

        def dispose():
            disposed[0] = True
            detector.cancel()

        return CompositeDisposable(source_sub, timer_sub, Disposable(dispose))

    return reactivex.create(subscribe)
