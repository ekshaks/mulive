import re

from .logging_utils import log_text_block, monitor_log
from .events import ClientTranscriptMessage
from .stream_dsl import client_message_sink, expand_items, map_items
from .tts_providers import tts_sink
from .tts_providers.factory import TTSConfig, create_tts_provider

# Split after sentence/clause punctuation (plus any closing quotes/brackets)
# followed by whitespace so every pipeline can share the same phrase boundary.
SPOKEN_PHRASE_BOUNDARY = re.compile(r'(?<=[,.!?;:])(?:["\')\]]+)?\s+')


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


_TTS_PROVIDER_ALIASES = {"kokoro": "kokoro_fastapi"}


def _resolve_tts_provider_name(provider_name):
    """Canonicalize a TTS provider identifier (translating legacy aliases).

    Treat the legacy ``"kokoro"`` value as an alias for the FastAPI provider.
    """
    return _TTS_PROVIDER_ALIASES.get(provider_name, provider_name)


def make_tts_provider(provider_name, output_mode, audio_output):
    """Adapt legacy helper arguments to the canonical TTS factory."""
    return create_tts_provider(
        TTSConfig(provider=_resolve_tts_provider_name(provider_name), output=output_mode),
        audio_output=audio_output,
    )


# Compatibility for existing app entrypoints and tests; new code uses the
# public factory above.
_make_tts_provider = make_tts_provider


def add_kokoro_tts(
    stream,
    audio_output,
    turn_signals,
    subs,
    mode,
    provider="kokoro_fastapi",
    name_prefix="kokoro",
):
    """Attach a TTS sink to ``stream`` using the legacy helper name."""
    return add_tts(
        stream,
        audio_output,
        turn_signals,
        subs=subs,
        mode=mode,
        provider=provider,
        name_prefix=name_prefix,
    )


def add_tts(
    stream,
    audio_output,
    turn_signals,
    subs,
    mode,
    provider="kokoro_fastapi",
    name_prefix="tts",
):
    """Attach a TTS sink to ``stream``.

    ``mode`` picks the output surface (``local`` = server speaker via
    sounddevice, ``browser`` = the session's audio output). ``provider``
    picks the synthesis backend (``kokoro_fastapi``, ``kokoro_onnx``, or
    ``piper``).
    """
    if mode is None:
        return

    output_mode = {"local": "local", "browser": "webrtc"}.get(mode)
    if output_mode is None:
        raise ValueError(f"Unknown TTS mode: {mode}")
    output = audio_output if output_mode == "webrtc" else None

    canonical_provider = _resolve_tts_provider_name(provider)
    tts_provider = make_tts_provider(canonical_provider, output_mode, output)
    monitor_log(
        f"tts sink attached provider_input={provider} "
        f"provider_effective={canonical_provider} mode={mode} name_prefix={name_prefix}"
    )

    if canonical_provider in _NON_STREAMING_TTS_PROVIDERS:
        # Whole-clip providers pay all their synthesis time before any audio
        # comes out. Split into short phrases so first-audio latency stays
        # bounded per request; streaming providers (piper, kokoro_fastapi)
        # already emit PCM incrementally and don't need this.
        stream = stream | expand_items(split_spoken_phrases, name=f"{name_prefix}_phrase_split")

    name = f"{name_prefix}_{canonical_provider}_{mode}_tts"
    stream.to(
        tts_sink(tts_provider, interrupts=turn_signals, name=name),
        name=name,
        subs=subs,
    )
    return tts_provider
