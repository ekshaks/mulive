"""Deepgram prerecorded STT adapter for completed VAD segments."""

import asyncio
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from ..events import TranscriptEvent
from ..logging_utils import monitor_time

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


class DeepgramSTT:
    """Transcribe 16 kHz mono PCM audio with Deepgram's prerecorded API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "nova-2",
        language: str = "en",
        smart_format: bool = True,
        timeout_s: float = 20.0,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required when stt.provider is 'deepgram'.")
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.timeout_s = timeout_s

    def transcribe_turn(self, segment: np.ndarray) -> str:
        pcm = _pcm16_bytes(segment)
        if not pcm:
            return ""
        query = urlencode(
            {
                "model": self.model,
                "language": self.language,
                "smart_format": str(self.smart_format).lower(),
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
            }
        )
        request = Request(
            f"{DEEPGRAM_LISTEN_URL}?{query}",
            data=pcm,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/raw",
            },
            method="POST",
        )
        started_at = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Deepgram request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Deepgram connection failed: {exc.reason}") from exc
        finally:
            audio_seconds = len(pcm) / (16000 * 2)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            monitor_time(
                "stt",
                "transcribe",
                elapsed_ms / 1000,
                provider="deepgram",
                model=self.model,
                audio_s=f"{audio_seconds:.2f}",
            )
        return _transcript_from_response(payload)


def _pcm16_bytes(segment: np.ndarray) -> bytes:
    samples = np.asarray(segment).reshape(-1)
    if samples.dtype != np.int16:
        if np.issubdtype(samples.dtype, np.floating):
            samples = np.clip(samples, -1.0, 1.0) * 32767
        samples = samples.astype(np.int16)
    return samples.tobytes()


def _transcript_from_response(payload: dict) -> str:
    try:
        return payload["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Deepgram response did not contain a transcript.") from exc


def deepgram_stt(
    name: str = "deepgram_stt",
    model: str = "nova-2",
    language: str = "en",
    smart_format: bool = True,
    timeout_s: float = 20.0,
    on_status=None,
    **kwargs,
):
    """Create an async final-transcript stage backed by Deepgram REST STT."""

    from ..stream_dsl import async_map_stage

    client = DeepgramSTT(
        model=model,
        language=language,
        smart_format=smart_format,
        timeout_s=timeout_s,
        **kwargs,
    )

    async def transcribe_turn(segment):
        context = getattr(segment, "context", None)
        samples = getattr(segment, "samples", segment)
        try:
            text = await asyncio.to_thread(client.transcribe_turn, samples)
        except Exception as exc:
            if on_status:
                on_status("error", {"model": model, "reason": str(exc)})
            return TranscriptEvent(text="", is_final=True, context=context)
        if on_status:
            on_status("ready", {"model": model})
        return TranscriptEvent(text=text, is_final=True, context=context)

    return async_map_stage(transcribe_turn, name=name)
