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
  Hugging Face on first use into ``$MULIVE_CACHE_DIR/piper/`` (or
  ``$XDG_CACHE_HOME/mulive/piper/``). If set to an explicit path, that
  path must exist (no download attempted).
* ``PIPER_CONFIG_PATH``  — path to ``<voice>.onnx.json`` (default:
  ``<PIPER_MODEL_PATH>.json``).

Air-gapped deployments: pre-place the voice at either the env-overridden
path or the cache directory to skip the network round-trip.
"""

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import numpy as np

from ..audio_output import AudioChunk
from ..logging_utils import monitor_log, monitor_time


# Default voice — small (~60 MB), permissively licensed, ships with the
# piper1-gpl catalog and is well-tested on 1-vCPU CPU boxes.
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"

# Immutable Hugging Face revision for the default voice. Bump this revision
# and both hashes together when upgrading the bundled default.
_PIPER_VOICE_REVISION = "1162a9173d0ce503555aed757976b7a9912eae4c"
_PIPER_VOICE_URL_BASE = (
    f"https://huggingface.co/rhasspy/piper-voices/resolve/{_PIPER_VOICE_REVISION}/"
    "en/en_US/lessac/medium"
)
_PIPER_ASSETS = {
    "model": {
        "filename": f"{DEFAULT_PIPER_VOICE}.onnx",
        "sha256": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    },
    "config": {
        "filename": f"{DEFAULT_PIPER_VOICE}.onnx.json",
        "sha256": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    },
}


def _piper_cache_dir() -> Path:
    """Return (and create) the directory used to cache Piper voice assets."""
    root = os.environ.get("MULIVE_CACHE_DIR")
    if root is None:
        root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        cache = Path(root) / "mulive" / "piper"
    else:
        cache = Path(root) / "piper"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_piper_asset(kind: str, dest: Path) -> Path:
    """Download a pinned Piper asset atomically and verify its SHA-256."""
    asset = _PIPER_ASSETS[kind]
    url = f"{_PIPER_VOICE_URL_BASE}/{asset['filename']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Piper asset not found locally; downloading from {url} -> {dest}")
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=60) as response, tmp.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        actual = digest.hexdigest()
        if actual != asset["sha256"]:
            raise RuntimeError(
                f"Downloaded Piper {kind} asset has sha256 {actual}, expected "
                f"{asset['sha256']}. Refusing to cache it."
            )
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _ensure_piper_asset(kind: str) -> Path:
    """Locate (and if needed download) the Piper voice file for ``kind``.

    ``kind`` is either ``"model"`` (``.onnx``) or ``"config"``
    (``.onnx.json``).

    Resolution order — mirrors :func:`mulive.core.turndet._find_silero_onnx_model`:

    1. Explicit env override (``PIPER_MODEL_PATH`` / ``PIPER_CONFIG_PATH``).
       If set, the file must already exist — we never auto-download to a
       user-specified path.
    2. Cached copy at ``$MULIVE_CACHE_DIR/piper/`` (or the XDG cache).
    3. Fresh download of the default voice from the pinned Hugging Face
       revision into the cache directory, with SHA-256 verification.
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

    asset = _PIPER_ASSETS[kind]
    filename = asset["filename"]
    cached = _piper_cache_dir() / filename
    if cached.exists():
        actual = hashlib.sha256(cached.read_bytes()).hexdigest()
        if actual != asset["sha256"]:
            raise RuntimeError(
                f"Cached Piper {kind} asset has sha256 {actual}, expected {asset['sha256']}. "
                f"Remove {cached} and retry, or set an explicit trusted path."
            )
        return cached
    return _download_piper_asset(kind, cached)


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
