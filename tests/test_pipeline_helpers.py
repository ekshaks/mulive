import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import reactivex
from reactivex.subject import Subject

from mulive.core.audio_output import AudioChunk
from mulive.core.pipeline_helpers import add_text_sinks, add_tts, non_empty_text, to_user
from mulive.core.stream_dsl import Stream, SubGroup
from mulive.core.tts_providers import TTSConfig


class FakeAudioOutput:
    async def write(self, chunk: AudioChunk) -> None:
        del chunk

    async def wait_until_drained(self) -> None:
        pass

    def clear(self) -> None:
        pass


class PipelineTextHelperTests(unittest.TestCase):
    def test_non_empty_text_drops_empty_and_whitespace_values(self):
        received = []
        stream = Stream.source(reactivex.from_iterable([None, "", "  ", "hello"]))

        (stream | non_empty_text()).observable.subscribe(received.append)

        self.assertEqual(received, ["hello"])

    def test_sensitive_transcript_can_skip_server_log_without_skipping_client(self):
        session = SimpleNamespace(send_to_client=Mock())
        subs = SubGroup()
        stream = Stream.source(reactivex.just("private verification claims"))

        with patch("mulive.core.pipeline_helpers.log_text_block") as log_text:
            add_text_sinks(
                stream,
                session,
                role="user",
                subs=subs,
                log=False,
            )

        log_text.assert_not_called()
        sent = session.send_to_client.call_args.args[0]
        self.assertEqual(sent.content, "private verification claims")
        subs.dispose()

    def test_to_user_without_tts_is_a_terminal_text_stage(self):
        session = SimpleNamespace(send_to_client=Mock())
        subs = SubGroup()

        result = Stream.source(reactivex.just("hello")) | to_user(
            session=session, role="user", subs=subs, log=False,
        )

        self.assertIsNone(result)
        self.assertEqual(session.send_to_client.call_args.args[0].content, "hello")
        subs.dispose()


class PipelineTTSHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_tts_attaches_existing_provider(self):
        subs = SubGroup()
        provider = SimpleNamespace(speak=AsyncMock())

        result = add_tts(
            Stream.source(Subject()),
            provider,
            TTSConfig(provider="piper", mode="browser"),
            Stream.source(Subject()),
            subs=subs,
        )

        self.assertIsNone(result)
        subs.dispose()
        await asyncio.sleep(0)

    async def test_to_user_uses_config_for_text_and_tts(self):
        session = SimpleNamespace(audio_output=FakeAudioOutput(), send_to_client=Mock())
        subs = SubGroup()
        source = Subject()
        signals = Stream.source(Subject())
        config = TTSConfig(provider="piper", mode="browser")
        provider = SimpleNamespace(speak=AsyncMock(), aclose=AsyncMock())

        result = Stream.source(source) | to_user(
            session=session, role="assistant", subs=subs,
            tts=config, tts_provider=provider, interrupts=signals,
        )
        source.on_next("Hello there.")
        for _ in range(100):
            if provider.speak.await_count:
                break
            await asyncio.sleep(0.01)

        self.assertIsNone(result)
        self.assertEqual(session.send_to_client.call_args.args[0].content, "Hello there.")
        provider.speak.assert_awaited_once()
        await subs.aclose()
        provider.aclose.assert_not_awaited()

    def test_to_user_requires_config_and_provider_together(self):
        session = SimpleNamespace(send_to_client=Mock())
        subs = SubGroup()
        with self.assertRaisesRegex(ValueError, "supplied together"):
            to_user(session=session, role="assistant", subs=subs, tts=TTSConfig())

    def test_tts_config_translates_legacy_output(self):
        self.assertEqual(TTSConfig(provider="piper", output="webrtc").mode, "browser")


if __name__ == "__main__":
    unittest.main()
