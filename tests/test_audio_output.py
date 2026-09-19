"""Contracts for transport-neutral assistant audio output."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from mulive.core.audio_output import AudioChunk, AudioOutput
from mulive.core.session import SessionContext
from mulive.core.tts_providers.factory import TTSConfig, create_tts_provider
from mulive.core.tts_providers import kokoro_fastapi, kokoro_onnx, piper
from mulive.core.webrtc_audio import AssistantAudioTrack
from mulive.core.websocket_audio import WebSocketPCMOutput
from mulive.server.setup_tracks import pc_session_setup


class FakeAudioOutput:
    def __init__(self):
        self.chunks = []

    async def write(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)

    async def wait_until_drained(self) -> None:
        pass

    def clear(self) -> None:
        self.chunks.clear()


class FakePeer:
    def __init__(self, *args, **kwargs):
        self.connectionState = "new"
        self.handlers = {}

    def addTrack(self, track) -> None:
        self.outbound_track = track

    def on(self, event, callback=None):
        if callback is None:
            return lambda handler: self.on(event, handler)
        self.handlers[event] = callback
        return callback


class AudioChunkTests(unittest.TestCase):
    def test_accepts_explicit_pcm16_metadata(self):
        samples = np.array([1, -1, 2, -2], dtype=np.int16)

        chunk = AudioChunk(samples, sample_rate=24000, channels=2)

        self.assertIs(chunk.samples, samples)
        self.assertEqual(chunk.sample_rate, 24000)
        self.assertEqual(chunk.channels, 2)

    def test_rejects_non_array_and_non_pcm16_samples(self):
        with self.assertRaisesRegex(TypeError, "numpy array"):
            AudioChunk([1, 2], sample_rate=16000)
        with self.assertRaisesRegex(TypeError, "signed 16-bit PCM"):
            AudioChunk(np.array([0.1], dtype=np.float32), sample_rate=16000)

    def test_rejects_non_flat_samples(self):
        with self.assertRaisesRegex(ValueError, "flat interleaved"):
            AudioChunk(np.zeros((2, 2), dtype=np.int16), sample_rate=16000)

    def test_rejects_invalid_rate_channels_and_alignment(self):
        samples = np.zeros(4, dtype=np.int16)
        for rate in (True, 1.5):
            with self.subTest(sample_rate=rate):
                with self.assertRaises(TypeError):
                    AudioChunk(samples, sample_rate=rate)
        with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
            AudioChunk(samples, sample_rate=0)
        for channels in (True, 1.5):
            with self.subTest(channels=channels):
                with self.assertRaises(TypeError):
                    AudioChunk(samples, sample_rate=16000, channels=channels)
        with self.assertRaisesRegex(ValueError, "channels must be positive"):
            AudioChunk(samples, sample_rate=16000, channels=0)
        with self.assertRaisesRegex(ValueError, "complete channel frames"):
            AudioChunk(np.zeros(3, dtype=np.int16), sample_rate=16000, channels=2)


class AudioOutputTests(unittest.TestCase):
    def test_structural_protocol_accepts_transport_implementation(self):
        self.assertIsInstance(FakeAudioOutput(), AudioOutput)

    def test_session_exposes_transport_neutral_output(self):
        output = FakeAudioOutput()
        session = SessionContext(
            pc=None,
            audio_output=output,
            data_channels={},
            audio_input=None,
            video_input=None,
            client_input=None,
            main_loop=SimpleNamespace(),
        )

        self.assertIs(session.audio_output, output)
        self.assertIs(session.assistant_audio_track, output)

    def test_factory_receives_audio_output_without_peer_connection(self):
        output = FakeAudioOutput()

        provider = create_tts_provider(
            TTSConfig(provider="piper", output="webrtc"),
            audio_output=output,
        )

        self.assertIs(provider.audio_track, output)

    def test_factory_rejects_transport_object_without_audio_contract(self):
        with self.assertRaisesRegex(TypeError, "AudioOutput"):
            create_tts_provider(
                TTSConfig(provider="piper", output="webrtc"),
                audio_output=SimpleNamespace(assistant_audio_track=FakeAudioOutput()),
            )

    def test_factory_keeps_previous_output_keywords(self):
        for keyword in ("pcm_output", "audio_track"):
            with self.subTest(keyword=keyword):
                output = FakeAudioOutput()
                provider = create_tts_provider(
                    TTSConfig(provider="piper", output="webrtc"),
                    **{keyword: output},
                )
                self.assertIs(provider.audio_track, output)

    def test_factory_rejects_conflicting_output_keywords(self):
        with self.assertRaisesRegex(ValueError, "only one audio output"):
            create_tts_provider(
                TTSConfig(provider="piper", output="webrtc"),
                audio_output=FakeAudioOutput(),
                audio_track=FakeAudioOutput(),
            )

class WebRTCAudioOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_concrete_outputs_satisfy_drain_contract(self):
        async def send_bytes(_payload):
            pass

        websocket = WebSocketPCMOutput(send_bytes, asyncio.Lock())
        self.assertIsInstance(websocket, AudioOutput)
        self.assertIsInstance(AssistantAudioTrack(), AudioOutput)
        await websocket.close()

    async def test_track_accepts_typed_audio_chunk(self):
        output = AssistantAudioTrack(sample_rate=16000)
        samples = np.array([1, -1, 2, -2], dtype=np.int16)

        await output.write(AudioChunk(samples, sample_rate=16000))

        queued = output._queue.get_nowait()
        np.testing.assert_array_equal(queued, samples)

    async def test_track_rejects_multichannel_chunk(self):
        output = AssistantAudioTrack(sample_rate=16000)
        chunk = AudioChunk(np.zeros(4, dtype=np.int16), 16000, channels=2)

        with self.assertRaisesRegex(ValueError, "requires mono"):
            await output.write(chunk)

    async def test_webrtc_setup_injects_track_as_session_audio_output(self):
        async def run_session(session):
            return None

        with (
            patch("mulive.server.setup_tracks.RTCPeerConnection", FakePeer),
            patch("mulive.server.setup_tracks.AssistantAudioTrack", FakeAudioOutput),
        ):
            peer = pc_session_setup(run_session, config={})
            await asyncio.sleep(0)

        self.assertIs(peer.session_context.audio_output, peer.outbound_track)
        self.assertIsInstance(peer.session_context.audio_output, AudioOutput)


class ProviderAudioChunkTests(unittest.IsolatedAsyncioTestCase):
    async def test_kokoro_fastapi_provider_closes_http_client(self):
        provider = object.__new__(kokoro_fastapi.KokoroFastApiTTSProvider)
        provider.client = SimpleNamespace(close=AsyncMock())

        await provider.aclose()

        provider.client.close.assert_awaited_once()

    async def test_kokoro_fastapi_writes_typed_chunk(self):
        output = FakeAudioOutput()

        async def synthesize(text, interrupt_event, callback, client=None):
            await callback(np.array([1, -1], dtype=np.int16), 24000)

        with patch.object(kokoro_fastapi, "_tts_kokoro_stream_chunks", synthesize):
            await kokoro_fastapi.tts_kokoro_to_track_async(
                "hello", asyncio.Event(), output
            )

        self.assertEqual(output.chunks[0].sample_rate, 24000)
        self.assertEqual(output.chunks[0].samples.dtype, np.int16)

    async def test_kokoro_onnx_writes_typed_chunk(self):
        output = FakeAudioOutput()
        synthesized = np.array([0.25, -0.25], dtype=np.float32)

        with patch.object(
            kokoro_onnx,
            "_create_kokoro_audio",
            AsyncMock(return_value=(synthesized, 24000)),
        ):
            await kokoro_onnx._stream_kokoro_to_track(
                "hello", asyncio.Event(), output, "voice", 1.0, "en-us"
            )

        self.assertEqual(output.chunks[0].sample_rate, 24000)
        self.assertEqual(output.chunks[0].samples.dtype, np.int16)

    async def test_piper_writes_typed_chunk(self):
        output = FakeAudioOutput()

        async def synthesize(text, interrupt_event, callback):
            await callback(np.array([1, -1], dtype=np.int16), 22050)

        with patch.object(piper, "_stream_piper_pcm", synthesize):
            await piper._stream_piper_to_track("hello", asyncio.Event(), output)

        self.assertEqual(output.chunks[0].sample_rate, 22050)
        self.assertEqual(output.chunks[0].samples.dtype, np.int16)


if __name__ == "__main__":
    unittest.main()
