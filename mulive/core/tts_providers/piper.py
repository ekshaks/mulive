"""In-process Piper TTS provider.

Piper (``OHF-Voice/piper1-gpl``, PyPI ``piper-tts``) is a small,
GPL-3.0 ONNX TTS. On 1-vCPU CPU boxes its real-time factor is ~0.1–0.3
for the *medium* voices, and unlike ``kokoro_onnx`` it streams PCM as
it synthesizes — so first-audio latency is a small fraction of the
utterance length instead of the whole clip's synthesis time. That
directly kills the TTS gaps we saw with ``kokoro_onnx`` on the AWS
1-vCPU box.

Two output modes:

* ``local``  — plays synthesized PCM on the server speaker via
  ``sounddevice``. Same shape as the other providers.
* ``webrtc`` — writes 20 ms int16 PCM frames into an outbound WebRTC
  audio track (``pc.assistant_audio_track``) for browser playback.

Model / config file paths are configurable via env:

* ``PIPER_MODEL_PATH``   — path to ``<voice>.onnx``. If unset the
  default ``en_US-lessac-medium`` voice is used and auto-fetched from
  Hugging Face on first use into ``$XDG_CACHE_HOME/mulive/piper/``. If
  set to an explicit path, that path must exist (no download attempted).
* ``PIPER_CONFIG_PATH``  — path to ``<voice>.onnx.json`` (default:
  ``<PIPER_MODEL_PATH>.json``).

Air-gapped deployments: pre-place the voice at either the env-overridden
path or the cache directory to skip the network round-trip.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio_output import AudioChunk
from ..logging_utils import monitor_log, monitor_time


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default voice — small (~60 MB), permissively licensed, ships with the
# piper1-gpl catalog and is well-tested on 1-vCPU CPU boxes.
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"

# Upstream layout on the rhasspy/piper-voices Hugging Face repo mirrors
# <lang>/<locale>/<voice>/<quality>/<voice>-<quality>.onnx[.json].
_PIPER_VOICE_URL_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium"
)


def _piper_cache_dir() -> Path:
    """Return (and create) the directory used to cache Piper voice assets."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache = Path(base) / "mulive" / "piper"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_piper_asset(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` atomically.

    Writes to ``dest.with_suffix('.part')`` first so a partial download
    from a Ctrl-C never presents as a valid cached asset. Unlike the
    Silero fetcher we don't pin a SHA-256 here — Piper's HF catalog is
    versioned by voice/quality name, not by revision, and shipping a
    hash per voice would require pinning every voice we might ever add.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Piper asset not found locally; downloading from {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _ensure_piper_asset(kind: str) -> Path:
    """Locate (and if needed download) the Piper voice file for ``kind``.

    ``kind`` is either ``"model"`` (``.onnx``) or ``"config"``
    (``.onnx.json``).

    Resolution order — mirrors :func:`mulive.core.turndet._find_silero_onnx_model`:

    1. Explicit env override (``PIPER_MODEL_PATH`` / ``PIPER_CONFIG_PATH``).
       If set, the file must already exist — we never auto-download to a
       user-specified path.
    2. Pre-existing file at ``ext/piper/<voice>.onnx[.json]`` under the
       repo root (deployment-friendly, matches the old manual layout).
    3. Cached copy at ``$XDG_CACHE_HOME/mulive/piper/<voice>.onnx[.json]``.
    4. Fresh download of the default voice from the pinned Hugging Face
       URL into the cache dir.
    """
    suffix = ".onnx" if kind == "model" else ".onnx.json"
    env_var = "PIPER_MODEL_PATH" if kind == "model" else "PIPER_CONFIG_PATH"
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value)
        if not path.exists():
            raise FileNotFoundError(f"{env_var} does not exist: {path}")
        return path
    if kind == "config":
        # If only PIPER_MODEL_PATH is set, mirror the model's directory
        # for the config file — same convention as the manual layout.
        model_override = os.environ.get("PIPER_MODEL_PATH")
        if model_override:
            path = Path(model_override).with_suffix(".onnx.json")
            if not path.exists():
                raise FileNotFoundError(
                    f"Piper config file not found next to PIPER_MODEL_PATH: {path}"
                )
            return path

    filename = f"{DEFAULT_PIPER_VOICE}{suffix}"
    bundled = PROJECT_ROOT / "ext" / "piper" / filename
    if bundled.exists():
        return bundled
    cached = _piper_cache_dir() / filename
    if cached.exists():
        return cached
    return _download_piper_asset(f"{_PIPER_VOICE_URL_BASE}/{filename}", cached)


def _model_path() -> Path:
    return _ensure_piper_asset("model")


def _config_path() -> Path:
    return _ensure_piper_asset("config")


_voice = None


def get_voice():
    """Load the Piper voice on first use and cache it process-wide."""
    global _voice
    if _voice is None:
        from piper import PiperVoice

        _voice = PiperVoice.load(str(_model_path()), str(_config_path()))
    return _voice


def _synthesize_pcm_bytes(text: str) -> tuple[list[bytes], int]:
    """Synthesize ``text`` into a list of raw int16 PCM byte chunks.

    Piper 1.x (``piper1-gpl``) exposes ``PiperVoice.synthesize(text)`` as
    a generator of ``AudioChunk`` dataclasses. Each chunk carries a
    ``sample_rate`` and raw little-endian int16 PCM bytes via
    ``audio_int16_bytes``. We drain the generator in a worker thread so
    the event loop stays responsive; each yielded chunk is a natural
    streaming boundary (typically one sentence).
    """
    voice = get_voice()
    chunks: list[bytes] = []
    sample_rate: Optional[int] = None
    for chunk in voice.synthesize(text):
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        chunks.append(chunk.audio_int16_bytes)
    if sample_rate is None:
        # ``synthesize`` yielded nothing (empty text) — fall back to the
        # voice's configured sample rate so downstream framing math works.
        sample_rate = voice.config.sample_rate
    return chunks, sample_rate


async def _stream_piper_pcm(text, interrupt_event, on_audio_block):
    """Delegate int16 PCM blocks from Piper to ``on_audio_block``."""
    if interrupt_event.is_set():
        return
    started_at = time.perf_counter()
    loop = asyncio.get_event_loop()
    chunks, sample_rate = await loop.run_in_executor(None, _synthesize_pcm_bytes, text)
    monitor_time(
        "tts",
        "synthesize",
        time.perf_counter() - started_at,
        provider="piper",
    )
    frame_samples = max(1, int(sample_rate * 0.02))  # 20 ms frames

    first = True
    for chunk_bytes in chunks:
        if interrupt_event.is_set():
            monitor_log("tts provider=piper event=interrupted")
            return
        block = np.frombuffer(chunk_bytes, dtype=np.int16)
        for start in range(0, len(block), frame_samples):
            if interrupt_event.is_set():
                monitor_log("tts provider=piper event=interrupted")
                return
            frame = block[start : start + frame_samples]
            await on_audio_block(frame, sample_rate)
            if first:
                first = False
                monitor_log("tts provider=piper event=first_pcm")


def warm_up() -> None:
    """Force-load the Piper voice and pay the graph-init cost.

    First ``PiperVoice.load`` + first synthesize on a fresh process is
    ~200–800 ms on a 1-vCPU box. Doing it before the server accepts
    connections keeps that off the first user utterance.
    """
    started_at = time.perf_counter()
    voice = get_voice()
    for _ in voice.synthesize("hi"):
        pass
    print(f"Piper warmed up in {time.perf_counter() - started_at:.2f} s")


async def _play_piper_local(text, interrupt_event):
    """Play a Piper stream on the server's default output device."""
    import sounddevice as sd

    voice = get_voice()
    stream = sd.OutputStream(
        samplerate=voice.config.sample_rate,
        channels=1,
        dtype="int16",
    )
    stream.start()

    loop = asyncio.get_event_loop()

    async def write_local(block, sample_rate):
        await loop.run_in_executor(None, stream.write, block)

    try:
        await _stream_piper_pcm(text, interrupt_event, write_local)
    finally:
        stream.stop()
        stream.close()


async def _stream_piper_to_track(text, interrupt_event, audio_track):
    """Write Piper PCM into an outbound WebRTC audio track."""
    first_track_write = True

    async def write_track(block, sample_rate):
        nonlocal first_track_write
        if first_track_write:
            first_track_write = False
            monitor_log("tts provider=piper event=first_pcm_to_webrtc_track")
        await audio_track.write(AudioChunk(block, sample_rate))

    await _stream_piper_pcm(text, interrupt_event, write_track)


class PiperTTSProvider:
    """In-process Piper TTS provider.

    Parameters
    ----------
    output:
        ``"local"`` writes to the server speaker via ``sounddevice``.
        ``"webrtc"`` writes 20 ms PCM blocks into ``audio_track``.
    audio_track:
        Outbound aiortc audio track — required when ``output="webrtc"``.
    """

    def __init__(
        self,
        output: str = "local",
        audio_track: Optional[Any] = None,
    ):
        if output not in {"local", "webrtc"}:
            raise ValueError(f"Unknown PiperTTSProvider output: {output}")
        if output == "webrtc" and audio_track is None:
            raise ValueError("audio_track is required for PiperTTSProvider(output='webrtc')")
        self.output = output
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        if self.output == "local":
            await _play_piper_local(text, interrupt_event)
            return
        await _stream_piper_to_track(text, interrupt_event, self.audio_track)

    def clear_output(self) -> None:
        """Best-effort clear of any queued audio in the outbound track."""
        if (
            self.output == "webrtc"
            and self.audio_track is not None
            and hasattr(self.audio_track, "clear")
        ):
            self.audio_track.clear()
