"""Speech in, text out — with the real Whisper backend and real spoken audio.

Two things are checked here, both of which had gone wrong in the chess coach:

1. A configured backend whose package is not installed must say so, loudly, once.
   It used to raise inside a worker thread, one utterance at a time, where the
   error was swallowed and turned into an empty transcript — so the app looked
   deaf and nothing anywhere said why.
2. A single failed transcription must not end the transcript stream. The stream is
   the controller's only speech input and is never resubscribed, so one error
   would leave the session deaf for the rest of the game.

The audio is genuinely spoken: macOS ``say`` writes a clip, ``afconvert`` turns it
into the 16 kHz mono PCM the pipeline uses, and the real faster-whisper model
transcribes it. No mocks, no fakes.
"""

import asyncio
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from reactivex.subject import Subject

# Production sets this from `stt.offline: true`; do the same so a cached model is
# used and the test never reaches for the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    check_stt_provider,
    stt,
    turn_detector,
)
from mulive.core.stt.whisper import WhisperSTT, require_backend
from mulive.core.stt import STTConfig

SPOKEN_WORDS = "Knight to f 3"
MODEL_SIZE = "small"


def spoken_audio(words: str) -> np.ndarray:
    """Speak ``words`` with the system voice and return 16 kHz mono samples.

    Args:
        words: What to say.

    Returns:
        Signed 16-bit samples, exactly as the pipeline delivers them.

    Raises:
        unittest.SkipTest: When the machine has no ``say``/``afconvert``.
    """
    if not shutil.which("say") or not shutil.which("afconvert"):
        raise unittest.SkipTest("needs macOS 'say' and 'afconvert' to speak the test clip")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "clip.aiff"
        target = Path(folder) / "clip.wav"
        subprocess.run(["say", "-o", str(source), words], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(source), str(target)],
            check=True,
        )
        with wave.open(str(target)) as clip:
            return np.frombuffer(clip.readframes(clip.getnframes()), dtype=np.int16)


def cached_model_or_skip() -> None:
    """Skip the test when the faster-whisper weights are not on this machine.

    Only a missing download is a reason to skip. Anything else — a broken backend, a
    bad argument, an import error — is a real failure and is allowed through, so a
    regression can never quietly turn into a skipped test.

    Raises:
        unittest.SkipTest: When the weights are not in the local cache.
    """
    try:
        WhisperSTT(STTConfig(variant=MODEL_SIZE))
    except OSError as exc:
        # huggingface_hub raises LocalEntryNotFoundError (an OSError) when a model is
        # not cached and downloads are off.
        raise unittest.SkipTest(f"faster-whisper '{MODEL_SIZE}' weights not cached: {exc}")


class BackendAvailabilityTests(unittest.TestCase):
    """A backend that cannot run must be reported when the app loads."""

    def test_an_installed_backend_passes(self):
        check_stt_provider("faster_whisper")

    def test_an_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            check_stt_provider("whisperish")

    def test_deepgram_needs_no_local_package(self):
        check_stt_provider("deepgram")

    @unittest.skipIf(
        importlib.util.find_spec("mlx_whisper") is not None,
        "mlx-whisper is installed here, so its absence cannot be tested",
    )
    def test_a_missing_backend_names_the_package_and_the_way_out(self):
        for call in (lambda: require_backend("mlx"), lambda: check_stt_provider("mlx")):
            with self.assertRaises(RuntimeError) as caught:
                call()
            message = str(caught.exception)
            self.assertIn("mlx-whisper", message)
            self.assertIn("faster_whisper", message)

    @unittest.skipIf(
        importlib.util.find_spec("mlx_whisper") is not None,
        "mlx-whisper is installed here, so its absence cannot be tested",
    )
    def test_building_the_stage_fails_instead_of_going_quiet(self):
        with self.assertRaises(RuntimeError):
            stt(STTConfig(provider="mlx", variant=MODEL_SIZE))


class SpokenAudioTests(unittest.IsolatedAsyncioTestCase):
    """Real speech through the real turn detector and the real model."""

    async def transcripts(
        self, *clips, expected: int = 1, on_status=None, **config_values
    ) -> list[str]:
        """Push each clip through turn detection and STT, and collect the text.

        Every clip goes through the *same* subscription, so a second clip really does
        test that the stream survived whatever the first one did.

        Args:
            *clips: 16 kHz mono int16 audio arrays, one per utterance.
            expected: How many transcripts to wait for.
            on_status: Optional status callback for the STT stage.
            **config_values: Overrides for the STT configuration.

        Returns:
            The text of every final transcript, in order.
        """
        audio = Subject()
        options = {"provider": "faster_whisper", "variant": MODEL_SIZE}
        options.update(config_values)
        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True, silence_timeout=0.01, poll_interval=0.005
        )
        texts: list[str] = []
        subs = SubGroup()
        (turn.value | stt(STTConfig.from_mapping(options), on_status=on_status)).to(
            lambda observable: observable.subscribe(lambda event: texts.append(event.text)),
            subs=subs,
        )
        try:
            for index, clip in enumerate(clips):
                audio.on_next(clip)
                # Wait for this clip's transcript before speaking again, so the
                # transcripts stay in step with the clips.
                await self.wait_for(texts, index + 1)
            await self.wait_for(texts, expected)
        finally:
            subs.dispose()
        return texts

    async def wait_for(self, texts: list, count: int) -> None:
        """Wait until ``texts`` holds ``count`` items, or give up.

        The budget is deliberately longer than the stage's own timeout, so a slow but
        valid transcription cannot be mistaken for a failure.

        Args:
            texts: The collected transcripts.
            count: How many are expected.
        """
        for _ in range(600):
            if len(texts) >= count:
                return
            await asyncio.sleep(0.05)

    async def test_a_spoken_move_arrives_as_text(self):
        cached_model_or_skip()
        texts = await self.transcripts(spoken_audio(SPOKEN_WORDS))
        self.assertEqual(len(texts), 1)
        self.assertIn("knight", texts[0].lower())

    async def test_a_failed_transcription_leaves_the_stream_alive(self):
        cached_model_or_skip()
        statuses = []
        # A timeout this small cannot be met, so the first segment really does fail —
        # no patched objects needed to reach the failure path. The second segment
        # proves the stream is still there afterwards.
        texts = await self.transcripts(
            spoken_audio(SPOKEN_WORDS),
            spoken_audio(SPOKEN_WORDS),
            expected=2,
            timeout_seconds=0.001,
            on_status=lambda result, data: statuses.append((result, data)),
        )
        self.assertEqual(texts, ["", ""])
        self.assertEqual([result for result, _ in statuses], ["error", "error"])
        self.assertIn("timed out", statuses[0][1]["reason"])

    async def test_a_status_callback_that_throws_cannot_deafen_the_session(self):
        # The real callback writes to a data channel that may have just closed.
        cached_model_or_skip()

        def explode(result, data):
            raise RuntimeError("data channel closed")

        texts = await self.transcripts(spoken_audio(SPOKEN_WORDS), on_status=explode)
        self.assertEqual(len(texts), 1)
        self.assertIn("knight", texts[0].lower())


if __name__ == "__main__":
    unittest.main()
