"""Transport-neutral formation of committed voice turns."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, replace

import numpy as np

@dataclass(frozen=True)
class VoiceTurn:
    """One voice turn from its first speech frame through cancellation."""

    id: str
    pcm16: bytes = b""
    generation: int = 0
    cancelled: asyncio.Event = field(default_factory=asyncio.Event, compare=False)

    @property
    def turn_id(self) -> str:
        """Compatibility alias for protocol payloads that use ``turn_id``."""
        return self.id

    @property
    def samples(self) -> np.ndarray:
        """PCM16 samples for STT stages that operate on arrays."""
        return np.frombuffer(self.pcm16, dtype=np.int16)

    def with_pcm16(self, pcm16: bytes) -> "VoiceTurn":
        _validate_pcm16(pcm16)
        return replace(self, pcm16=pcm16)

    def with_generation(self, generation: int) -> "VoiceTurn":
        return replace(self, generation=generation)


@dataclass(frozen=True)
class SpeechStarted:
    """Barge-in trigger emitted as soon as a new user turn starts."""

    turn: VoiceTurn

    @property
    def turn_id(self) -> str:
        return self.turn.id


class VoiceInput:
    """Async input boundary yielding speech starts and completed voice turns."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[SpeechStarted | VoiceTurn] = asyncio.Queue()

    async def started(self, turn: VoiceTurn) -> None:
        await self._events.put(SpeechStarted(turn))

    async def completed(self, turn: VoiceTurn) -> None:
        await self._events.put(turn)

    def __aiter__(self):
        return self

    async def __anext__(self) -> SpeechStarted | VoiceTurn:
        return await self._events.get()


class PTTTurnSource:
    """One explicitly started and committed PCM16 capture at a time."""

    def __init__(self, voice_input: VoiceInput, *, max_samples: int = 16_000 * 60) -> None:
        self.voice_input = voice_input
        self.max_samples = max_samples
        self.turn: VoiceTurn | None = None
        self._chunks: list[bytes] = []
        self._samples = 0

    @property
    def turn_id(self) -> str | None:
        return self.turn.id if self.turn is not None else None

    async def start(self, turn_id: str) -> None:
        if self.turn is not None:
            raise ValueError("capture turn already active")
        self.turn = VoiceTurn(turn_id)
        self._chunks = []
        self._samples = 0
        await self.voice_input.started(self.turn)

    def write(self, payload: bytes) -> None:
        if self.turn is None:
            raise ValueError("audio outside capture turn")
        _validate_pcm16(payload)
        samples = len(payload) // 2
        if self._samples + samples > self.max_samples:
            raise ValueError("turn exceeds 60 seconds")
        self._chunks.append(payload)
        self._samples += samples

    async def commit(self, turn_id: str) -> None:
        if self.turn is None or turn_id != self.turn.id:
            raise ValueError("commit outside capture turn")
        turn = self.turn
        pcm16 = b"".join(self._chunks)
        self._reset()
        if not pcm16:
            raise ValueError("empty pcm16")
        await self.voice_input.completed(turn.with_pcm16(pcm16))

    def cancel(self, turn_id: str | None = None) -> None:
        if turn_id is not None and (self.turn is None or turn_id != self.turn.id):
            return
        if self.turn is not None:
            self.turn.cancelled.set()
        self._reset()

    def _reset(self) -> None:
        self.turn = None
        self._chunks = []
        self._samples = 0


class VADTurnSource:
    """Continuous PCM16 source that emits a turn after VAD silence."""

    def __init__(self, voice_input: VoiceInput, is_speech, *, silence_timeout: float = 1.0, max_samples: int = 16_000 * 60) -> None:
        self.voice_input = voice_input
        self.is_speech = is_speech
        self.silence_timeout = silence_timeout
        self.max_samples = max_samples
        self.turn: VoiceTurn | None = None
        self._chunks: list[bytes] = []
        self._samples = 0
        self._last_speech = 0.0
        self._timer: asyncio.Task | None = None
        self._analysis = b""
        self._analysis_samples = 1_600

    @property
    def turn_id(self) -> str | None:
        return self.turn.id if self.turn is not None else None

    async def write(self, payload: bytes) -> None:
        _validate_pcm16(payload)
        self._analysis += payload
        window_bytes = self._analysis_samples * 2
        while len(self._analysis) >= window_bytes:
            window, self._analysis = self._analysis[:window_bytes], self._analysis[window_bytes:]
            samples = np.frombuffer(window, dtype="<i2")
            if self.is_speech(samples):
                if self.turn is None:
                    self.turn = VoiceTurn(uuid.uuid4().hex)
                    await self.voice_input.started(self.turn)
                    self._timer = asyncio.create_task(self._flush_after_silence(), name="voice-vad-flush")
                if self._samples + samples.size > self.max_samples:
                    await self._flush()
                    return
                self._chunks.append(window)
                self._samples += samples.size
                self._last_speech = time.monotonic()

    async def _flush_after_silence(self) -> None:
        try:
            while self.turn is not None:
                await asyncio.sleep(self.silence_timeout)
                if self.turn is not None and time.monotonic() - self._last_speech >= self.silence_timeout:
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self) -> None:
        turn, pcm16 = self.turn, b"".join(self._chunks)
        self.turn = None
        self._chunks = []
        self._samples = 0
        self._analysis = b""
        if turn is not None and pcm16:
            await self.voice_input.completed(turn.with_pcm16(pcm16))

    def cancel(self, _turn_id: str | None = None) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.turn is not None:
            self.turn.cancelled.set()
        self.turn = None
        self._chunks = []
        self._samples = 0


def _validate_pcm16(payload: bytes) -> None:
    if not payload or len(payload) % 2:
        raise ValueError("invalid pcm16")
