"""Speech-to-text provider adapters."""

from .config import STTConfig
from .deepgram import DeepgramSTT
from .pinned import PinnedWhisper
from .whisper import WhisperSTT, whisper_stt

__all__ = ["STTConfig", "DeepgramSTT", "PinnedWhisper", "WhisperSTT", "whisper_stt"]
