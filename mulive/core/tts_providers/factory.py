from dataclasses import dataclass
from typing import Any, Literal, Optional

from ..audio_output import AudioOutput


@dataclass
class TTSConfig:
    provider: Literal["kokoro_fastapi", "kokoro_onnx", "piper", "gemini"] = "kokoro_fastapi"
    output: Literal["local", "webrtc"] = "local"
    voice: Optional[str] = None
    model: Optional[str] = None


def create_tts_provider(
    config: TTSConfig,
    audio_output: Optional[Any] = None,
    *,
    pcm_output: Optional[Any] = None,
    audio_track: Optional[Any] = None,
):
    """Create a provider, accepting the two previous output keywords."""
    outputs = [item for item in (audio_output, pcm_output, audio_track) if item is not None]
    if outputs and any(item is not outputs[0] for item in outputs[1:]):
        raise ValueError("Provide only one audio output")
    if audio_output is None:
        audio_output = pcm_output if pcm_output is not None else audio_track
    output_mode = "webrtc" if config.output == "webrtc" else "local"
    pcm_providers = {"kokoro_fastapi", "kokoro_onnx", "piper"}
    if (
        output_mode == "webrtc"
        and config.provider in pcm_providers
        and not isinstance(audio_output, AudioOutput)
    ):
        raise TypeError("WebRTC TTS requires an AudioOutput implementation")
    if config.provider == "kokoro_fastapi":
        from .kokoro_fastapi import KokoroFastApiTTSProvider

        return KokoroFastApiTTSProvider(mode=output_mode, audio_track=audio_output)
    if config.provider == "kokoro_onnx":
        from .kokoro_onnx import KokoroOnnxTTSProvider

        return KokoroOnnxTTSProvider(
            voice=config.voice or "af_sarah",
            output=output_mode,
            audio_track=audio_output,
        )
    if config.provider == "piper":
        from .piper import PiperTTSProvider

        return PiperTTSProvider(output=output_mode, audio_track=audio_output)
    if config.provider == "gemini":
        from .gemini import GeminiTTSProvider

        return GeminiTTSProvider(
            model=config.model or "gemini-2.5-flash-preview-tts",
            voice=config.voice or "Kore",
        )
    raise ValueError(f"Unknown TTS provider: {config.provider}")
