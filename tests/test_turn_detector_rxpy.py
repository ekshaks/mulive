import asyncio
import unittest
import warnings

import numpy as np
from reactivex.subject import Subject

from mulive.core.stream_dsl import Stream, SubGroup, turn_detector
from mulive.core.turn import SpeechStarted


class TurnDetectorTests(unittest.IsolatedAsyncioTestCase):
    """The VAD poll timer is scheduled on the asyncio loop, so these tests
    subscribe and wait inside a running loop."""

    async def test_named_outputs_share_one_audio_subscription_and_dispose(self):
        audio = Subject()
        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
            silence_timeout=0.01,
            poll_interval=0.005,
        )
        segments = []
        values = []
        signals = []
        subs = SubGroup()

        segment_sub = turn.audio.to(
            lambda observable: observable.subscribe(segments.append),
            subs=subs,
        )
        signal_sub = turn.started.to(
            lambda observable: observable.subscribe(signals.append),
            subs=subs,
        )
        value_sub = turn.value.to(
            lambda observable: observable.subscribe(values.append),
            subs=subs,
        )

        self.assertEqual(len(audio.observers), 1)
        samples = np.zeros(1_600, dtype=np.int16)
        samples[:3] = [1, 2, 3]
        audio.on_next(samples)
        await asyncio.sleep(0.04)

        self.assertEqual(len(signals), 1)
        self.assertIsInstance(signals[0], SpeechStarted)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].pcm16, segments[0].tobytes())
        np.testing.assert_array_equal(
            segments[0],
            samples,
        )

        segment_sub.dispose()
        self.assertEqual(len(audio.observers), 1)
        signal_sub.dispose()
        value_sub.dispose()
        self.assertEqual(len(audio.observers), 0)

    async def test_old_named_outputs_warn_and_point_to_new_outputs(self):
        audio = Subject()
        turn = Stream.source(audio) | turn_detector(is_speech_fn=lambda _chunk: True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(turn.segments, turn.audio)
            self.assertIs(turn.turns, turn.value)
            self.assertIs(turn.signals, turn.started)
        self.assertEqual([str(item.message) for item in caught], [
            "turn.segments is deprecated; use turn.audio instead",
            "turn.turns is deprecated; use turn.value instead",
            "turn.signals is deprecated; use turn.started instead",
        ])

    async def test_completion_flushes_buffered_speech(self):
        audio = Subject()
        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
            silence_timeout=10,
            poll_interval=1,
        )
        segments = []
        completed = []
        subs = SubGroup()

        turn.audio.to(
            lambda observable: observable.subscribe(
                on_next=segments.append,
                on_completed=lambda: completed.append(True),
            ),
            subs=subs,
        )

        audio.on_next(np.array([4, 5], dtype=np.int16))
        audio.on_completed()

        self.assertEqual(len(segments), 1)
        self.assertEqual(completed, [True])
        self.assertEqual(len(audio.observers), 0)
        subs.dispose()

    async def test_subscribe_requires_running_loop(self):
        audio = Subject()
        observable = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
        )

        def subscribe_without_loop():
            with self.assertRaisesRegex(RuntimeError, "running asyncio"):
                observable.audio.to(
                    lambda obs: obs.subscribe(lambda _item: None),
                )

        await asyncio.to_thread(subscribe_without_loop)


if __name__ == "__main__":
    unittest.main()
