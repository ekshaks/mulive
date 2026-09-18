"""Transport-neutral source for captured mono PCM16 audio."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .audio_output import AudioChunk


@runtime_checkable
class AudioInput(Protocol):
    """Accept captured audio from a transport adapter."""

    def write(self, chunk: AudioChunk) -> None: ...
    def close(self) -> None: ...


class SubjectAudioInput:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Expose typed input while preserving the existing reactive stream API."""

    def __init__(self, subject: Any, *, sample_rate: int = 16_000):
        self._subject = subject
        self.sample_rate = sample_rate
        self._closed = False

    def write(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise RuntimeError("audio input is closed")
        _validate_capture_chunk(chunk, self.sample_rate)
        self._subject.on_next(chunk.samples)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._subject.on_completed()

    def subscribe(self, *args, **kwargs):
        return self._subject.subscribe(*args, **kwargs)


class BufferedAudioInput:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Bounded typed capture buffer for a committed push-to-talk turn."""

    def __init__(self, *, sample_rate: int = 16_000, max_samples: int):
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self._chunks: list[np.ndarray] = []
        self._sample_count = 0
        self._closed = False

    def write(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise RuntimeError("audio input is closed")
        _validate_capture_chunk(chunk, self.sample_rate)
        if chunk.samples.size == 0:
            raise ValueError("empty pcm16")
        if self._sample_count + chunk.samples.size > self.max_samples:
            raise ValueError("turn exceeds 60 seconds")
        self._chunks.append(chunk.samples.copy())
        self._sample_count += chunk.samples.size

    def pcm16(self) -> bytes:
        return b"".join(chunk.astype("<i2", copy=False).tobytes() for chunk in self._chunks)

    def close(self) -> None:
        self._closed = True
        self._chunks.clear()
        self._sample_count = 0


def _validate_capture_chunk(chunk: AudioChunk, sample_rate: int) -> None:
    if chunk.channels != 1:
        raise ValueError("captured audio must be mono")
    if chunk.sample_rate != sample_rate:
        raise ValueError(f"captured audio must be {sample_rate} Hz")
