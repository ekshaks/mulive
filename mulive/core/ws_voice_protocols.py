"""Loopback WebSocket protocol adapters for the shared voice turn engine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import inspect
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI
from .audio_output import SpeechResult
from .voice_engine import run_voice_turn
from .stt.pinned import PinnedWhisper
from .token_signing import decode_json, encode_json, sign, signature_matches
from .turn_source import PTTTurnSource, SpeechStarted, VADTurnSource, VoiceInput, VoiceTurn
from .turndet import _build_is_speech
from .tts_providers.kokoro_fastapi import _kokoro_fastapi_base_url
from .websocket_audio import WebSocketPCMOutput

PROTOCOL = "mulive.voice.v1"
PCM16_SAMPLE_RATE = 16_000
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class VoicePrincipal:
    subject: str
    display_name: str


class VoiceHandler(Protocol):
    """Generic app-owned voice pipeline supplied to a transport mulive."""

    async def wait_ready(self) -> None: ...

    async def run(self, pcm16, turn, is_current, emit, audio_output) -> None: ...

    async def aclose(self) -> None: ...


class VoiceTokenStore:
    """Short-lived single-use tokens; process-local replay prevention is intentional for MVP."""

    def __init__(self, secret: str, ttl_seconds: int = 300):
        if len(secret.encode()) < 32:
            raise ValueError("MULIVE_AUTH_COOKIE_SECRET must be at least 32 bytes")
        self.secret = secret.encode()
        self.ttl_seconds = ttl_seconds
        self.used: dict[str, int] = {}

    def issue(self, principal: VoicePrincipal) -> str:
        payload = {"sub": principal.subject, "name": principal.display_name, "aud": PROTOCOL, "exp": int(time.time()) + self.ttl_seconds, "jti": secrets.token_urlsafe(16)}
        encoded = encode_json(payload)
        return f"{encoded}.{sign(self.secret, encoded)}"

    def consume(self, token: str | None) -> VoicePrincipal | None:
        try:
            now = int(time.time())
            self.used = {jti: exp for jti, exp in self.used.items() if exp >= now}
            encoded, signature = (token or "").split(".", 1)
            if not signature_matches(self.secret, encoded, signature):
                return None
            payload = decode_json(encoded)
            if payload["aud"] != PROTOCOL or int(payload["exp"]) < now or payload["jti"] in self.used:
                return None
            self.used[payload["jti"]] = int(payload["exp"])
            return VoicePrincipal(payload["sub"], payload["name"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


class STT_LLM_TTS_Flow:
    """Provider configuration for the transport-neutral voice turn engine."""

    def __init__(self, *, config: dict | None = None):
        config = config or {}
        stt_config = config.get("stt") or {}
        tts_config = config.get("tts") or {}
        models_config = config.get("models") or {}
        self.stt = PinnedWhisper(
            mode=stt_config.get("provider") or os.getenv("MULIVE_VOICE_STT_MODE", "faster_whisper"),
            model_size=stt_config.get("model_size") or os.getenv("MULIVE_VOICE_STT_MODEL", "tiny"),
            language=stt_config.get("language", "en"),
        )
        self.stt_timeout_seconds = float(stt_config.get("timeout_seconds") or os.getenv("MULIVE_VOICE_STT_TIMEOUT_SECONDS", "30"))
        self.stt_load_timeout_seconds = float(stt_config.get("load_timeout_seconds") or os.getenv("MULIVE_VOICE_STT_LOAD_TIMEOUT_SECONDS", "180"))
        if self.stt_timeout_seconds <= 0 or self.stt_load_timeout_seconds <= 0:
            raise ValueError("voice STT timeouts must be positive")
        self.llm_model = models_config.get("text") or os.getenv("MULIVE_VOICE_LLM_MODEL", "openai/gpt-oss-20b")
        self.tts_client = AsyncOpenAI(
            base_url=tts_config.get("base_url") or _kokoro_fastapi_base_url(),
            api_key="not-needed",
        )

    async def wait_ready(self) -> None:
        await asyncio.wait_for(self.stt.wait_ready(), timeout=self.stt_load_timeout_seconds)

    async def run(self, pcm16, turn, is_current, emit, audio_output) -> None:
        transcribe_turn = getattr(self.stt, "transcribe_turn", None) or self.stt.transcribe
        await run_voice_turn(pcm16, turn=turn, transcribe_turn=transcribe_turn, stt_timeout_seconds=self.stt_timeout_seconds, llm_model=self.llm_model, is_current=is_current, emit=emit, audio_output=audio_output, response_id_factory=lambda: secrets.token_urlsafe(12), tts_client=getattr(self, "tts_client", None))

    def close(self) -> None:
        self.stt.shutdown()

    async def aclose(self) -> None:
        self.close()
        await self.tts_client.close()


# Previous callers used this concise name for the default handler.
VoicePipeline = STT_LLM_TTS_Flow


class VoiceSession:
    """Mode-selected turn source plus shared engine and serialized output."""

    def __init__(self, principal: VoicePrincipal, pipeline: VoiceHandler | None, send_json, send_bytes):
        self.principal, self.pipeline = principal, pipeline
        self._send_json, self._send_bytes = send_json, send_bytes
        self._send_lock = asyncio.Lock()
        self._active_response_id: str | None = None
        self.audio_output = WebSocketPCMOutput(
            send_bytes,
            self._send_lock,
            on_format=self._announce_audio_format,
        )
        self.ready = False
        self.closed = asyncio.Event()
        self._ready_event = asyncio.Event()
        self.mode: str | None = None
        self.voice_input: VoiceInput | None = None
        self.source = None
        self._events_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._generation = 0
        self._response_turn: VoiceTurn | None = None
        self._response_turn_id: str | None = None
        self.sequence = 0
        self._handler_closed = False
        self._bind_handler()

    def _bind_handler(self) -> None:
        if self.pipeline is None:
            return
        bind_session = getattr(self.pipeline, "bind_session", None)
        if bind_session is not None:
            bind_session(self)

    def set_handler(self, handler: VoiceHandler) -> None:
        if self.pipeline is not None:
            raise RuntimeError("voice handler already configured")
        self.pipeline = handler
        self._bind_handler()

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    @property
    def turn_id(self) -> str | None:
        if self.source is not None and self.source.turn_id is not None:
            return self.source.turn_id
        return self._response_turn_id

    @property
    def task(self):
        return self._response_task

    async def handle_json(self, event: dict) -> None:
        kind = event.get("type")
        if not self.ready:
            if kind != "session.hello" or (event.get("protocol") or event.get("text")) != PROTOCOL:
                raise ValueError("hello required")
            mode = event.get("mode", "ptt")
            if mode not in {"ptt", "vad"}:
                raise ValueError("unknown voice mode")
            self.ready, self.mode, self.voice_input = True, mode, VoiceInput()
            self.source = PTTTurnSource(self.voice_input) if mode == "ptt" else VADTurnSource(self.voice_input, _voice_is_speech())
            self._events_task = asyncio.create_task(self._consume_source_events(), name="voice-turn-events")
            self._ready_event.set()
            await self.emit({"type": "session.ready", "protocol": PROTOCOL, "mode": mode, "subject": self.principal.subject})
            return
        if kind == "tts.speak":
            text = event.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("tts.speak text required")
            await self._start_tts_response(text.strip())
            return
        if kind == "tts.cancel":
            await self._cancel_response(emit_cancelled=True)
            return
        if self.mode == "ptt":
            if kind == "turn.start":
                turn_id = event.get("turn_id")
                if not isinstance(turn_id, str) or not turn_id:
                    raise ValueError("turn_id required")
                await self.source.start(turn_id)
            elif kind == "turn.commit":
                await self.source.commit(event.get("turn_id"))
            elif kind == "turn.cancel":
                cancelled_turn_id = event.get("turn_id")
                await self._cancel_pending_action(cancelled_turn_id)
                self.source.cancel(cancelled_turn_id)
                await self._cancel_response(emit_cancelled=False)
                if cancelled_turn_id:
                    await self.emit({"type": "turn.finished", "turn_id": cancelled_turn_id, "outcome": "cancelled"})
            else:
                if self.pipeline is None or not hasattr(self.pipeline, "handle_client_event"):
                    raise ValueError("unsupported client event")
                await self.pipeline.handle_client_event(event, self.emit)
        elif kind in {"turn.start", "turn.commit", "turn.cancel"}:
            raise ValueError("PTT events are invalid in vad mode")
        else:
            if self.pipeline is None or not hasattr(self.pipeline, "handle_client_event"):
                raise ValueError("unsupported client event")
            await self.pipeline.handle_client_event(event, self.emit)

    async def handle_pcm16(self, payload: bytes) -> None:
        if not self.ready or self.source is None:
            raise ValueError("audio before hello")
        result = self.source.write(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _consume_source_events(self) -> None:
        assert self.voice_input is not None
        try:
            async for event in self.voice_input:
                if isinstance(event, SpeechStarted):
                    await self._cancel_pending_action()
                    await self._cancel_response(emit_cancelled=True)
                else:
                    await self._start_response(event)
        except asyncio.CancelledError:
            pass

    async def _start_response(self, turn: VoiceTurn) -> None:
        await self._cancel_response(emit_cancelled=False)
        self._generation += 1
        generation = self._generation
        turn = turn.with_generation(generation)
        self._response_turn = turn
        self._response_turn_id = turn.turn_id
        await self.emit({"type": "turn.committed", "turn_id": turn.turn_id})
        self._response_task = asyncio.create_task(self._run(turn), name=f"voice:{turn.turn_id}")

    async def _start_tts_response(self, text: str) -> None:
        if self.pipeline is None or not hasattr(self.pipeline, "speak"):
            raise ValueError("TTS is unavailable for this voice session")
        await self._cancel_response(emit_cancelled=False)
        self._generation += 1
        turn_id = f"tts-{secrets.token_urlsafe(8)}"
        turn = VoiceTurn(id=turn_id, generation=self._generation)
        self._response_turn = turn
        self._response_turn_id = turn_id
        await self.emit({"type": "turn.committed", "turn_id": turn_id})
        self._response_task = asyncio.create_task(
            self._run_tts(text, turn), name=f"voice:{turn_id}"
        )

    async def _run(self, turn: VoiceTurn) -> None:
        try:
            if self.pipeline is None:
                raise RuntimeError("voice handler is not configured")
            await self.pipeline.run(turn.pcm16, turn, lambda: self._response_turn is turn and not turn.cancelled.is_set(), self.emit, self.audio_output)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._response_turn is turn:
                logger.exception("voice pipeline failed turn_id=%s", turn.turn_id)
                await self.emit({"type": "error", "turn_id": turn.turn_id, "text": f"pipeline failed: {type(exc).__name__}"})
                await self.emit({"type": "turn.finished", "turn_id": turn.turn_id, "outcome": "failed", "reason": type(exc).__name__})
        finally:
            if self._response_turn is turn:
                self._response_task = None
                self._response_turn_id = None
                self._response_turn = None

    async def _run_tts(self, text: str, turn: VoiceTurn) -> SpeechResult:
        try:
            result = await self._speak(text, turn)
            if self._response_turn is turn and result.audio_completed:
                await self.emit({"type": "turn.finished", "turn_id": turn.id, "outcome": "spoken"})
            return result
        except asyncio.CancelledError:
            return SpeechResult(False, False)
        except Exception as exc:
            if self._response_turn is turn:
                logger.exception("voice TTS failed turn_id=%s", turn.id)
                await self.emit({"type": "error", "turn_id": turn.id, "text": f"TTS failed: {type(exc).__name__}"})
                await self.emit({"type": "turn.finished", "turn_id": turn.id, "outcome": "failed", "reason": type(exc).__name__})
            return SpeechResult(False, False)
        finally:
            if self._response_turn is turn:
                self._response_task = None
                self._response_turn_id = None
                self._response_turn = None

    async def speak(self, text: str) -> SpeechResult:
        """Speak text in the active turn, for an embed transcript callback."""
        turn = self._response_turn
        if turn is None:
            raise RuntimeError("session.speak() requires an active voice turn")
        return await self._speak(text, turn)

    async def speak_standalone(self, text: str) -> SpeechResult:
        """Server-initiated speech without fabricating a client turn."""
        await self._start_tts_response(text)
        task = self._response_task
        if task is None:
            return SpeechResult(False, False)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Barge-in cancels the greeting response, not the session runner.
            return SpeechResult(False, False)

    async def _speak(self, text: str, turn: VoiceTurn) -> SpeechResult:
        if self.pipeline is None or not hasattr(self.pipeline, "speak"):
            raise RuntimeError("TTS is unavailable for this voice session")
        if turn.cancelled.is_set() or self._response_turn is not turn:
            return SpeechResult(False, False)
        response_id = secrets.token_urlsafe(12)
        self._active_response_id = response_id
        self.audio_output.begin_response()
        await self.emit({"type": "response.started", "turn_id": turn.id, "response_id": response_id})
        await self.emit({"type": "response.text", "turn_id": turn.id, "response_id": response_id, "text": text})
        # ``text_delivered`` means exactly this await returned successfully.
        try:
            await self.pipeline.speak(text, turn.cancelled, self.audio_output)
        except Exception:
            return SpeechResult(True, False)
        if turn.cancelled.is_set() or self._response_turn is not turn:
            return SpeechResult(True, False)
        try:
            await self.audio_output.wait_until_drained()
        except Exception:
            return SpeechResult(True, False)
        if self._response_turn is turn:
            await self.emit({"type": "response.finished", "turn_id": turn.id, "response_id": response_id})
        if self._active_response_id == response_id:
            self._active_response_id = None
        return SpeechResult(True, True)

    async def _announce_audio_format(self, sample_rate: int) -> None:
        """Send format metadata before the first PCM frame of a response."""
        if self._active_response_id is None or self._response_turn is None:
            return
        await self.emit(
            {
                "type": "response.audio",
                "turn_id": self._response_turn.id,
                "response_id": self._active_response_id,
                "sample_rate": sample_rate,
                "channels": 1,
                "format": "pcm_s16le",
            }
        )

    async def _cancel_response(self, *, emit_cancelled: bool) -> None:
        task, turn_id = self._response_task, self._response_turn_id
        if task is None:
            return
        self._generation += 1
        if self._response_turn is not None:
            self._response_turn.cancelled.set()
            self._response_turn = None
        task.cancel()
        self.audio_output.clear()
        self._active_response_id = None
        self._response_task = None
        self._response_turn_id = None
        if emit_cancelled and turn_id is not None:
            await self.emit({"type": "response.cancelled", "turn_id": turn_id})
            await self.emit({"type": "turn.finished", "turn_id": turn_id, "outcome": "cancelled"})

    async def _cancel_pending_action(self, turn_id: str | None = None) -> None:
        if self.pipeline is None:
            return
        cancel_pending = getattr(self.pipeline, "cancel_for_new_turn", None)
        if cancel_pending is not None:
            await cancel_pending(turn_id)

    async def cancel(self) -> None:
        self.closed.set()
        await self._cancel_pending_action()
        if self.source is not None:
            self.source.cancel()
        await self._cancel_response(emit_cancelled=False)
        if self._events_task is not None:
            self._events_task.cancel()
            try:
                await self._events_task
            except asyncio.CancelledError:
                pass
            self._events_task = None
        await self.audio_output.close()
        await self._close_handler()

    async def _close_handler(self) -> None:
        if self._handler_closed or self.pipeline is None:
            return
        self._handler_closed = True
        close = getattr(self.pipeline, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def emit(self, event: dict) -> None:
        self.sequence += 1
        event["sequence"] = self.sequence
        async with self._send_lock:
            await self._send_json(event)


def _voice_is_speech():
    return _build_is_speech(threshold=0.4, min_speech_duration_ms=100, min_silence_duration_ms=2000, speech_pad_ms=200, rate=PCM16_SAMPLE_RATE)
