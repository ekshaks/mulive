import asyncio
import time
from typing import Optional

from ..logging_utils import monitor_time


CHANNELS = 1
RATE = 24000
SAMPLE_WIDTH = 2


def tts_gemini(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    from google import genai
    from google.genai import types

    client = genai.Client()
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    monitor_time(
        "tts",
        "synthesize",
        time.perf_counter() - started_at,
        provider="gemini",
        model=model,
    )
    return response.candidates[0].content.parts[0].inline_data.data


def play_pcm_data(pcm_data, channels=CHANNELS, rate=RATE, sample_width=SAMPLE_WIDTH):
    import sounddevice as sd

    with sd.RawOutputStream(
        samplerate=rate,
        channels=channels,
        dtype=f"int{sample_width * 8}",
    ) as stream:
        stream.write(pcm_data)


def gemini_tts_play(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    play_pcm_data(tts_gemini(text, model, voice))


async def tts_gemini_stream(text, model="gemini-2.5-flash-preview-tts", voice="Kore"):
    from google import genai
    from google.genai import types
    loop = asyncio.get_event_loop()
    client = genai.Client()
    started_at = time.perf_counter()
    first_audio_at = None
    import sounddevice as sd

    with sd.RawOutputStream(samplerate=RATE, channels=CHANNELS, dtype="int16") as stream:
        try:
            stream_gen = await client.aio.models.generate_content_stream(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            async for event in stream_gen:
                if (
                    hasattr(event, "candidates")
                    and event.candidates
                    and event.candidates[0].content.parts
                    and event.candidates[0].content.parts[0].inline_data
                ):
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                        monitor_time(
                            "tts",
                            "first_audio",
                            first_audio_at - started_at,
                            provider="gemini",
                            model=model,
                        )
                    await loop.run_in_executor(
                        None, stream.write, event.candidates[0].content.parts[0].inline_data.data
                    )
        finally:
            monitor_time(
                "tts",
                "stream_complete",
                time.perf_counter() - started_at,
                provider="gemini",
                model=model,
                first_audio_received=first_audio_at is not None,
            )


class GeminiTTSProvider:
    def __init__(self, model: str = "gemini-2.5-flash-preview-tts", voice: str = "Kore"):
        self.model = model
        self.voice = voice

    async def speak(self, text: str, interrupt_event: asyncio.Event) -> None:
        if interrupt_event.is_set():
            return
        await tts_gemini_stream(text, model=self.model, voice=self.voice)
