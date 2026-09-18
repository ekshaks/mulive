"""Speech-to-text provider adapters."""

from .deepgram import DeepgramSTT
from .pinned import PinnedWhisper
from .whisper import WhisperSTT, whisper_stt

__all__ = ["DeepgramSTT", "PinnedWhisper", "WhisperSTT", "whisper_stt"]
