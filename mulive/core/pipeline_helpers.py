import re

from .logging_utils import log_text_block, monitor_log
from .events import ClientTranscriptMessage
from .stream_dsl import client_message_sink, expand_items, filter_items, map_items
from .tts_providers import tts_sink
from .tts_providers.factory import TTSConfig

# Split after sentence/clause punctuation (plus any closing quotes/brackets)
# followed by whitespace so every pipeline can share the same phrase boundary.
SPOKEN_PHRASE_BOUNDARY = re.compile(r'(?<=[,.!?;:])(?:["\')\]]+)?\s+')


def _has_non_empty_text(text):
    return bool(text and text.strip())


def non_empty_text(name="non_empty_text"):
    """Keep text values that contain a non-whitespace character."""
    return filter_items(_has_non_empty_text, name=name)


def split_spoken_phrases(text):
    """Split spoken text into punctuation-delimited phrases for TTS.

    Non-streaming TTS providers (kokoro_onnx) synthesize a whole clip per
    request, so first-audio latency equals the synthesis time of the first
    phrase — short phrases keep that bounded. Returns a tuple of non-empty
    phrases.
    """
    return tuple(
        phrase.strip()
        for phrase in SPOKEN_PHRASE_BOUNDARY.split(text or "")
        if phrase.strip()
    )


def _text_log_sink(title, max_chars=1600):
    def attach(observable):
        return observable.subscribe(lambda text: log_text_block(title, text, max_chars=max_chars))
    return attach


def add_text_sinks(
    stream,
    session,
    role,
    subs,
    log_title=None,
    max_log_chars=1600,
    log=True,
):
    """Send transcript text to the client, with optional server-side logging.

    Set ``log=False`` for authentication or other sensitive user input. Client
    delivery is unaffected.
    """
    title = log_title or ("ASSISTANT WRITTEN RESPONSE" if role == "assistant" else "USER MESSAGE")
    if role != "assistant" and log:
        stream.to(_text_log_sink(title, max_chars=max_log_chars), name=f"log_{role}_text", subs=subs)
    client_messages = stream | map_items(
        lambda text: ClientTranscriptMessage(role=role, content=str(text)),
        name=f"{role}_client_message",
    )
    client_messages.to(
        client_message_sink(session),
        name=f"client_{role}_message",
        subs=subs,
    )


# Providers that synthesize the whole clip before yielding audio: split the
# input into short phrases so first-audio latency stays bounded.
_NON_STREAMING_TTS_PROVIDERS = frozenset({"kokoro_onnx"})


def add_tts(
    stream,
    tts_provider,
    config: TTSConfig,
    turn_signals,
    subs,
    name_prefix="tts",
):
    """Attach an already-created provider to a text stream."""
    monitor_log(
        f"tts sink attached provider={config.provider} mode={config.mode} name_prefix={name_prefix}"
    )

    if config.provider in _NON_STREAMING_TTS_PROVIDERS:
        # Whole-clip providers pay all their synthesis time before any audio
        # comes out. Split into short phrases so first-audio latency stays
        # bounded per request; streaming providers (piper, kokoro_fastapi)
        # already emit PCM incrementally and don't need this.
        stream = stream | expand_items(split_spoken_phrases, name=f"{name_prefix}_phrase_split")

    name = f"{name_prefix}_{config.provider}_{config.mode}_tts"
    stream.to(
        tts_sink(tts_provider, interrupts=turn_signals, name=name),
        name=name,
        subs=subs,
    )


def to_user(*, session, role, subs, tts: TTSConfig | None = None, tts_provider=None, interrupts=None, log=True):
    """Terminal stream stage; the caller owns the TTS provider lifecycle."""
    if tts is not None and not isinstance(tts, TTSConfig):
        raise TypeError("tts must be a TTSConfig or None")
    if (tts is None) != (tts_provider is None):
        raise ValueError("tts and tts_provider must be supplied together")

    def attach(stream):
        add_text_sinks(stream, session, role=role, subs=subs, log=log)
        if tts is not None:
            add_tts(
                stream,
                tts_provider,
                tts,
                interrupts,
                subs=subs,
            )
        return None

    return attach
