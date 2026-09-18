"""In-process Kokoro TTS backed by the ONNX runtime.

Two output modes:

* ``local``   — plays the synthesized audio on the server speaker via
  ``sounddevice``. Same as before; used on desktop.
* ``webrtc``  — writes 20 ms PCM blocks into an outbound WebRTC audio track
  so a browser client hears the audio. Preferred on headless servers.
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


_KOKORO_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
_KOKORO_ASSETS = {
    "model": {
        "filename": "kokoro-v1.0.onnx",
        "sha256": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    },
    "voices": {
        "filename": "voices-v1.0.bin",
        "sha256": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    },
}

_kokoro = None


def _kokoro_cache_dir() -> Path:
    """Return the Kokoro cache under ``MULIVE_CACHE_DIR`` or the platform cache."""
    root = os.environ.get("MULIVE_CACHE_DIR")
    if root is None:
        root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(root) / "mulive" / "kokoro"
    return Path(root) / "kokoro"


def _asset_path(kind: str) -> Path:
    env_var = "KOKORO_MODEL_PATH" if kind == "model" else "KOKORO_VOICES_PATH"
    override = os.environ.get(env_var)
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{env_var} does not exist: {path}")
        return path
    return _kokoro_cache_dir() / _KOKORO_ASSETS[kind]["filename"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(kind: str, destination: Path) -> Path:
    asset = _KOKORO_ASSETS[kind]
    url = f"{_KOKORO_RELEASE_URL}/{asset['filename']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    print(f"Kokoro {kind} asset not found locally; downloading {url} -> {destination}")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        actual = digest.hexdigest()
        if actual != asset["sha256"]:
            raise RuntimeError(
                f"Downloaded Kokoro {kind} asset has sha256 {actual}, expected "
                f"{asset['sha256']}. Refusing to cache it."
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _ensure_asset(kind: str) -> Path:
    path = _asset_path(kind)
    if os.environ.get("KOKORO_MODEL_PATH" if kind == "model" else "KOKORO_VOICES_PATH"):
        return path
    if path.exists():
        actual = _sha256(path)
        expected = _KOKORO_ASSETS[kind]["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Cached Kokoro {kind} asset has sha256 {actual}, expected {expected}. "
                f"Remove {path} and retry, or set an explicit trusted path."
            )
        return path
    return _download_asset(kind, path)


def ensure_kokoro_assets() -> tuple[Path, Path]:
    """Resolve the pinned Kokoro model and voices, downloading on first use."""
    return _ensure_asset("model"), _ensure_asset("voices")


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        model_path, voices_path = ensure_kokoro_assets()
        _kokoro = Kokoro(str(model_path), str(voices_path))
    return _kokoro


def warm_up(voice: str = "af_sarah", lang: str = "en-us") -> None:
    """Force Kokoro to load its ONNX weights and synthesize one tiny clip.

    The first ``Kokoro.create`` call on a fresh process pays the ONNX
    session-warmup cost (graph optimization, weight allocation, and the
    eSpeak G2P initialization). On a 1-vCPU box that is 1–3 s and, when
    it happens on the first user utterance, shows up as an audible gap
    before the assistant starts speaking. Calling this once before the
    server accepts connections moves the cost off the hot path.
    """
    import time as _time
    started_at = _time.perf_counter()
    kokoro = get_kokoro()
    # A one-word phrase is enough to trip the full pipeline (G2P + inference)
    # without producing meaningful audio.
    kokoro.create("hi", voice=voice, speed=1.0, lang=lang)
    print(f"Kokoro warmed up in {_time.perf_counter() - started_at:.2f} s")


async def _create_kokoro_audio(text, voice="af_sarah", speed=1.0, lang="en-us"):
    """Synthesize ``text`` to a (samples, sample_rate) pair off the event loop."""
    loop = asyncio.get_event_loop()
    started_at = time.perf_counter()
    audio = await loop.run_in_executor(
        None,
        lambda: get_kokoro().create(text, voice=voice, speed=speed, lang=lang),
    )
    monitor_time(
        "tts",
        "synthesize",
        time.perf_counter() - started_at,
        provider="kokoro_onnx",
        voice=voice,
    )
    return audio


def _to_pcm16(samples):
    """Convert kokoro-onnx float32 samples in [-1, 1] to int16 PCM."""
    if samples.dtype == np.int16:
        return samples
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


async def _play_kokoro_local(text, interrupt_event, voice, speed, lang):
    """Play a synthesized clip on the server's default output device."""
    import sounddevice as sd

    samples, sr = await _create_kokoro_audio(text, voice=voice, speed=speed, lang=lang)
    chunk_size = int(sr * 0.5)
    loop = asyncio.get_event_loop()
    for start in range(0, len(samples), chunk_size):
        if interrupt_event.is_set():
            sd.stop()
            break
        chunk = samples[start : start + chunk_size]
        await loop.run_in_executor(None, lambda chunk=chunk: sd.play(chunk, sr, blocking=True))


async def _stream_kokoro_to_track(text, interrupt_event, audio_track, voice, speed, lang):
    """Synthesize ``text`` and push int16 PCM into the outbound WebRTC track.

    ``kokoro_onnx.Kokoro.create`` is not streaming, so first-audio latency is
    the whole-clip synthesis time. Keep each utterance short (single sentence
    or two) upstream to bound perceived latency.
    """
    samples, sr = await _create_kokoro_audio(text, voice=voice, speed=speed, lang=lang)
    if interrupt_event.is_set() or len(samples) == 0:
        return
    pcm = _to_pcm16(samples)
    frame_samples = max(1, int(sr * 0.02))  # 20 ms frames — the aiortc default
    first_write_at = None
    started_at = time.perf_counter()
    for start in range(0, len(pcm), frame_samples):
        if interrupt_event.is_set():
            monitor_log("tts provider=kokoro_onnx event=interrupted")
            break
        block = pcm[start : start + frame_samples]
        await audio_track.write(AudioChunk(block, sr))
        if first_write_at is None:
            first_write_at = time.perf_counter()
            monitor_log("tts provider=kokoro_onnx event=first_pcm_to_webrtc_track")
    if first_write_at is not None:
        monitor_time(
            "tts",
            "first_audio_track_write",
            first_write_at - started_at,
            provider="kokoro_onnx",
            voice=voice,
        )


class KokoroOnnxTTSProvider:
    """In-process Kokoro TTS provider (ONNX runtime).

    Parameters
    ----------
    voice, speed, lang:
        Voice pack, speech rate, and language tag passed to ``Kokoro.create``.
    output:
        ``"local"`` writes to the server speaker via ``sounddevice``.
        ``"webrtc"`` writes 20 ms PCM blocks into ``audio_track``.
    audio_track:
        Outbound aiortc audio track — required when ``output="webrtc"``.
    """

    def __init__(
        self,
        voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        output: str = "local",
        audio_track: Optional[Any] = None,
    ):
        if output not in {"local", "webrtc"}:
            raise ValueError(f"Unknown KokoroOnnxTTSProvider output: {output}")
        if output == "webrtc" and audio_track is None:
            raise ValueError("audio_track is required for KokoroOnnxTTSProvider(output='webrtc')")
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.output = output
        self.audio_track = audio_track

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        if self.output == "local":
            await _play_kokoro_local(text, interrupt_event, self.voice, self.speed, self.lang)
            return
        await _stream_kokoro_to_track(
            text, interrupt_event, self.audio_track, self.voice, self.speed, self.lang
        )

    def clear_output(self) -> None:
        """Best-effort clear of any queued audio in the outbound track."""
        if self.output == "webrtc" and self.audio_track is not None and hasattr(self.audio_track, "clear"):
            self.audio_track.clear()
