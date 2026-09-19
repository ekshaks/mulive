"""Independent STT and TTS flows for embedding Mulive in a host app."""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Awaitable, Callable, TYPE_CHECKING

import numpy as np

from .stt.deepgram import DeepgramSTT
from .stt.config import STTConfig
from .stt.pinned import PinnedWhisper
from .tts_providers.factory import TTSConfig, create_tts_provider
from .turndet import warm_up_vad

if TYPE_CHECKING:
    from .audio_output import AudioOutput
    from .turn import TurnContext
    from .ws_voice_protocols import VoiceSession


TranscriptCallback = Callable[[str, "VoiceSession"], Awaitable[None] | None]


class STTFlow:
    """Configuration-backed completed-turn transcription."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = STTConfig.from_mapping(config)
        self.timeout_seconds = self.config.timeout_seconds

        if self.config.provider in {"mlx", "faster_whisper"}:
            self.provider: PinnedWhisper | DeepgramSTT = PinnedWhisper(
                self.config,
            )
        elif self.config.provider == "deepgram":
            self.provider = DeepgramSTT(
                model=self.config.model or "nova-2",
                language=self.config.language,
                smart_format=bool(self.config.options.get("smart_format", True)),
                timeout_s=self.timeout_seconds,
            )
        else:
            raise ValueError(f"Unknown STT provider: {self.config.provider}")

    async def wait_ready(self) -> None:
        if isinstance(self.provider, PinnedWhisper):
            await self.provider.wait_ready()

    async def transcribe_turn(self, pcm16: bytes) -> str:
        samples = np.frombuffer(pcm16, dtype=np.int16)
        if isinstance(self.provider, PinnedWhisper):
            operation = self.provider.transcribe_turn(samples)
        else:
            operation = asyncio.to_thread(self.provider.transcribe_turn, samples)
        return await asyncio.wait_for(operation, timeout=self.timeout_seconds)

    async def aclose(self) -> None:
        if isinstance(self.provider, PinnedWhisper):
            self.provider.shutdown()


class TTSFlow:
    """Configuration-backed server synthesis for one session audio output."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))

    def create_speaker(self, audio_output: "AudioOutput"):
        if not self.enabled:
            raise RuntimeError("TTS is disabled by configuration")
        provider = str(self.config.get("provider") or "kokoro_onnx")
        if provider == "gemini":
            raise ValueError("Gemini TTS does not yet support embedded PCM output")
        return create_tts_provider(
            TTSConfig(
                provider=provider,
                mode="browser",
                voice=self.config.get("voice"),
                model=self.config.get("model"),
            ),
            audio_output=audio_output,
        )


class VoiceRuntime:
    """Application-scoped STT/TTS resources and session runner lifecycle."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        vad_config = dict(config.get("vad") or {})
        self.vad_backend = str(vad_config.get("backend") or "").strip().lower()
        if self.vad_backend and self.vad_backend not in {"torch", "onnx"}:
            raise ValueError("vad.backend must be 'torch' or 'onnx'")
        self.stt_flow = STTFlow(config.get("stt"))
        self.tts_flow = TTSFlow(config.get("tts"))

    async def prepare(self) -> None:
        if self.vad_backend:
            os.environ.setdefault("SILERO_BACKEND", self.vad_backend)
        await asyncio.gather(self.stt_flow.wait_ready(), asyncio.to_thread(warm_up_vad))

    def create_session_runner(self, handler_factory, *, greeting: str | None = None):
        async def run_session(session):
            if session.pipeline is None:
                session.set_handler(handler_factory())
            await session.wait_until_ready()
            if greeting is not None:
                await session.speak_standalone(greeting)
            await session.closed.wait()

        run_session.prepare = self.prepare
        run_session.close = self.aclose
        return run_session

    async def aclose(self) -> None:
        await self.stt_flow.aclose()


class EmbedVoiceHandler:
    """Per-session composition of independent STT and TTS flows."""

    def __init__(self, stt_flow: STTFlow, tts_flow: TTSFlow, on_transcript: TranscriptCallback | None) -> None:
        self.stt_flow = stt_flow
        self.tts_flow = tts_flow
        self.on_transcript = on_transcript
        self.session: VoiceSession | None = None
        self._speaker = None

    def bind_session(self, session: "VoiceSession") -> None:
        self.session = session
        if self.tts_flow.enabled:
            self._speaker = self.tts_flow.create_speaker(session.audio_output)

    async def wait_ready(self) -> None:
        await self.stt_flow.wait_ready()

    async def run(self, pcm16, context: "TurnContext", is_current, emit, _audio_output) -> None:
        text = await self.stt_flow.transcribe_turn(pcm16)
        if not is_current():
            return
        await emit({"type": "transcript.final", "turn_id": context.id, "text": text})
        if self.on_transcript is not None:
            if self.session is None:
                raise RuntimeError("embed voice handler is not bound to a session")
            result = self.on_transcript(text, self.session)
            if inspect.isawaitable(result):
                await result
        if is_current():
            await emit({"type": "turn.finished", "turn_id": context.id, "outcome": "transcribed"})

    async def speak(self, text: str, cancelled: asyncio.Event, _audio_output) -> None:
        if self._speaker is None:
            raise RuntimeError("embed voice handler is not bound to a session")
        await self._speaker.speak(text, cancelled)

    async def aclose(self) -> None:
        if self._speaker is not None:
            close = getattr(self._speaker, "aclose", None)
            if close is not None:
                await close()


class EmbedRuntime(VoiceRuntime):
    """Shared provider lifecycle and per-WebSocket handler creation."""

    def __init__(self, config: dict[str, Any] | None = None, on_transcript: TranscriptCallback | None = None) -> None:
        super().__init__(config)
        self.on_transcript = on_transcript

    async def wait_ready(self) -> None:
        await self.prepare()

    def create_handler(self) -> EmbedVoiceHandler:
        return EmbedVoiceHandler(self.stt_flow, self.tts_flow, self.on_transcript)

    def create_session_runner(self):
        return super().create_session_runner(self.create_handler)
