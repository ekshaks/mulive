"""Transport-neutral formation of committed voice turns."""

from __future__ import annotations

import asyncio

import numpy as np

from .turn import SpeechStarted, TurnEvent, VoiceTurn, validate_pcm16
from .turndet import TurnDetector


class VoiceInput:
    """Async input boundary yielding speech starts and completed voice turns."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[TurnEvent] = asyncio.Queue()

    async def started(self, turn_id: str) -> None:
        """Publish the start of a turn."""
        await self._events.put(SpeechStarted(turn_id))

    async def completed(self, turn: VoiceTurn) -> None:
        """Publish a completed turn."""
        await self._events.put(turn)

    def __aiter__(self):
        """Return this input as its asynchronous event iterator."""
        return self

    async def __anext__(self) -> TurnEvent:
        """Wait for and return the next input event."""
        return await self._events.get()


class PTTTurnSource:
    """One explicitly started and committed PCM16 capture at a time."""

    def __init__(self, voice_input: VoiceInput, *, max_samples: int = 16_000 * 60) -> None:
        self.voice_input = voice_input
        self.max_samples = max_samples
        self._turn_id: str | None = None
        self._chunks: list[bytes] = []
        self._samples = 0

    @property
    def turn_id(self) -> str | None:
        """Return the active push-to-talk turn identifier."""
        return self._turn_id

    async def start(self, turn_id: str) -> None:
        """Start a push-to-talk capture with the supplied identifier."""
        if self._turn_id is not None:
            raise ValueError("capture turn already active")
        self._turn_id = turn_id
        self._chunks = []
        self._samples = 0
        await self.voice_input.started(turn_id)

    def write(self, payload: bytes) -> None:
        """Append validated PCM16 bytes to the active capture."""
        if self._turn_id is None:
            raise ValueError("audio outside capture turn")
        validate_pcm16(payload)
        samples = len(payload) // 2
        if self._samples + samples > self.max_samples:
            raise ValueError("turn exceeds 60 seconds")
        self._chunks.append(payload)
        self._samples += samples

    async def commit(self, turn_id: str) -> None:
        """Publish and close the matching active capture."""
        if self._turn_id is None or turn_id != self._turn_id:
            raise ValueError("commit outside capture turn")
        committed_id = self._turn_id
        pcm16 = b"".join(self._chunks)
        self._reset()
        if not pcm16:
            raise ValueError("empty pcm16")
        await self.voice_input.completed(VoiceTurn(committed_id, pcm16))

    def cancel(self, turn_id: str | None = None) -> None:
        """Discard the active capture when its identifier matches."""
        if turn_id is not None and (self._turn_id is None or turn_id != self._turn_id):
            return
        self._reset()

    def _reset(self) -> None:
        self._turn_id = None
        self._chunks = []
        self._samples = 0


class WebSocketVADSource:
    """Adapt WebSocket PCM16 bytes to the canonical turn detector."""

    def __init__(
        self,
        voice_input: VoiceInput,
        is_speech,
        *,
        silence_timeout: float = 1.0,
        max_samples: int = 16_000 * 60,
    ) -> None:
        self.voice_input = voice_input
        self.detector = TurnDetector(
            is_speech,
            silence_timeout=silence_timeout,
            max_samples=max_samples,
        )
        self._timer: asyncio.Task | None = None

    @property
    def turn_id(self) -> str | None:
        """Return the active detected turn identifier."""
        return self.detector.turn_id

    async def write(self, payload: bytes) -> None:
        """Feed validated WebSocket PCM16 bytes into turn detection."""
        validate_pcm16(payload)
        await self._publish(self.detector.feed(np.frombuffer(payload, dtype="<i2")))

    async def _flush_after_silence(self) -> None:
        try:
            while self.detector.turn_id is not None:
                await asyncio.sleep(self.detector.silence_timeout)
                await self._flush()
        except asyncio.CancelledError:
            return

    async def _flush(self) -> None:
        await self._publish(self.detector.flush_if_ready())
        if self.detector.turn_id is None:
            self._timer = None

    async def _publish(self, events: list[TurnEvent]) -> None:
        for event in events:
            if isinstance(event, SpeechStarted):
                await self.voice_input.started(event.turn_id)
                if self._timer is None:
                    self._timer = asyncio.create_task(
                        self._flush_after_silence(), name="voice-vad-flush"
                    )
            else:
                await self.voice_input.completed(event)

    def cancel(self, _turn_id: str | None = None) -> None:
        """Cancel the flush timer and discard the active detected turn."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.detector.cancel()
