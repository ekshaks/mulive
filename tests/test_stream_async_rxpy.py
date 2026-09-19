import asyncio
import unittest

import numpy as np
from reactivex.subject import Subject

from mulive.core.events import TranscriptEvent
from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    async_map_stage,
    final_transcript_text,
    turn_detector,
)


class AsyncMapStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_serial_preserves_order(self):
        source = Subject()
        calls = []
        received = []
        subs = SubGroup()

        async def transform(value):
            calls.append(value)
            await asyncio.sleep(0.01 if value == 1 else 0)
            return value * 10

        output = Stream.source(source) | async_map_stage(transform)
        output.to(
            lambda observable: observable.subscribe(received.append),
            subs=subs,
        )

        source.on_next(1)
        source.on_next(2)
        source.on_completed()
        await asyncio.sleep(0.04)

        self.assertEqual(calls, [1, 2])
        self.assertEqual(received, [10, 20])
        subs.dispose()

    async def test_multiple_sinks_share_one_async_call(self):
        source = Subject()
        calls = []
        first = []
        second = []
        subs = SubGroup()

        async def transform(value):
            calls.append(value)
            await asyncio.sleep(0)
            return value.upper()

        output = Stream.source(source) | async_map_stage(transform)
        output.to(
            lambda observable: observable.subscribe(first.append),
            subs=subs,
        )
        output.to(
            lambda observable: observable.subscribe(second.append),
            subs=subs,
        )

        source.on_next("hello")
        await asyncio.sleep(0.01)

        self.assertEqual(calls, ["hello"])
        self.assertEqual(first, ["HELLO"])
        self.assertEqual(second, ["HELLO"])
        subs.dispose()

    async def test_disposal_cancels_in_flight_coroutine(self):
        source = Subject()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        subs = SubGroup()

        async def transform(_value):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        output = Stream.source(source) | async_map_stage(transform)
        output.to(
            lambda observable: observable.subscribe(),
            subs=subs,
        )

        source.on_next("work")
        await asyncio.wait_for(started.wait(), timeout=1)
        subs.dispose()
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    async def test_vad_to_async_transcript_across_scheduler_thread(self):
        audio = Subject()
        subs = SubGroup()
        received = []

        async def transcribe_turn(turn):
            return TranscriptEvent(
                text=str(int(turn.samples.sum())),
                is_final=True,
            )

        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
            silence_timeout=0.01,
            poll_interval=0.005,
        )
        transcripts = turn.value | async_map_stage(transcribe_turn)
        final_text = transcripts | final_transcript_text()
        final_text.to(
            lambda observable: observable.subscribe(received.append),
            subs=subs,
        )

        samples = np.zeros(1_600, dtype=np.int16)
        samples[:3] = [10, 20, 12]
        audio.on_next(samples)
        await asyncio.sleep(0.06)

        self.assertEqual(received, ["42"])
        subs.dispose()

    async def test_stage_cleanup_runs_once_after_last_sink(self):
        source = Subject()
        cleanup_calls = []
        subs = SubGroup()

        async def transform(value):
            return value

        output = Stream.source(source) | async_map_stage(
            transform,
            on_dispose=lambda: cleanup_calls.append(True),
        )
        first = output.to(
            lambda observable: observable.subscribe(),
            subs=subs,
        )
        second = output.to(
            lambda observable: observable.subscribe(),
            subs=subs,
        )

        first.dispose()
        self.assertEqual(cleanup_calls, [])
        second.dispose()
        self.assertEqual(cleanup_calls, [True])

    async def test_errors_propagate(self):
        source = Subject()
        errors = []
        subs = SubGroup()

        async def transform(_value):
            raise ValueError("bad input")

        output = Stream.source(source) | async_map_stage(transform)
        output.to(
            lambda observable: observable.subscribe(on_error=errors.append),
            subs=subs,
        )

        source.on_next("bad")
        await asyncio.sleep(0.01)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        subs.dispose()

    async def test_latest_cancels_stale_work(self):
        source = Subject()
        cancelled = []
        received = []
        subs = SubGroup()

        async def transform(value):
            try:
                if value == "old":
                    await asyncio.sleep(10)
                return value
            except asyncio.CancelledError:
                cancelled.append(value)
                raise

        output = Stream.source(source) | async_map_stage(
            transform,
            concurrency="latest",
        )
        output.to(
            lambda observable: observable.subscribe(received.append),
            subs=subs,
        )

        source.on_next("old")
        await asyncio.sleep(0)
        source.on_next("new")
        await asyncio.sleep(0.02)

        self.assertEqual(cancelled, ["old"])
        self.assertEqual(received, ["new"])
        subs.dispose()

    async def test_drop_ignores_new_input_while_busy(self):
        source = Subject()
        release = asyncio.Event()
        calls = []
        received = []
        subs = SubGroup()

        async def transform(value):
            calls.append(value)
            if value == "first":
                await release.wait()
            return value

        output = Stream.source(source) | async_map_stage(
            transform,
            concurrency="drop",
        )
        output.to(
            lambda observable: observable.subscribe(received.append),
            subs=subs,
        )

        source.on_next("first")
        await asyncio.sleep(0)
        source.on_next("second")
        release.set()
        await asyncio.sleep(0.02)

        self.assertEqual(calls, ["first"])
        self.assertEqual(received, ["first"])
        subs.dispose()


if __name__ == "__main__":
    unittest.main()
