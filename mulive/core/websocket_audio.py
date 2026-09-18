"""Bounded, ordered PCM16 delivery for WebSocket voice responses."""

from __future__ import annotations

import asyncio

from .audio_output import AudioChunk


class WebSocketPCMOutput:
    def __init__(
        self,
        send_bytes,
        send_lock: asyncio.Lock,
        *,
        max_chunks: int = 64,
        on_format=None,
    ) -> None:
        self._send_bytes = send_bytes
        self._send_lock = send_lock
        self._queue: asyncio.Queue[tuple[int, bytes, int]] = asyncio.Queue(maxsize=max_chunks)
        self._generation = 0
        self._on_format = on_format
        self._announced_sample_rate: int | None = None
        self._worker = asyncio.create_task(self._send_loop(), name="voice-websocket-pcm")

    def begin_response(self) -> None:
        """Require the next audio chunk to announce its sample rate."""
        self._announced_sample_rate = None

    async def write(self, chunk: AudioChunk) -> None:
        if chunk.channels != 1:
            raise ValueError("WebSocket PCM output requires mono chunks")
        if chunk.sample_rate != self._announced_sample_rate:
            self._announced_sample_rate = chunk.sample_rate
            if self._on_format is not None:
                await self._on_format(chunk.sample_rate)
        generation = self._generation
        await self._queue.put((generation, chunk.samples.astype("<i2", copy=False).tobytes(), chunk.sample_rate))

    def clear(self) -> None:
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                return

    async def wait_until_drained(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        self.clear()
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    async def _send_loop(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            generation, payload, sample_rate = await self._queue.get()
            try:
                if generation == self._generation:
                    async with self._send_lock:
                        if generation == self._generation:
                            await self._send_bytes(payload)
                    duration = (len(payload) / 2) / sample_rate
                    deadline += duration
                    now = loop.time()
                    # Preserve the absolute clock across ordinary scheduler
                    # overshoot, but discard it after a real idle gap.
                    if deadline < now - 0.25:
                        deadline = now
                    await asyncio.sleep(max(0, deadline - loop.time()))
                else:
                    deadline = loop.time()
            finally:
                self._queue.task_done()
