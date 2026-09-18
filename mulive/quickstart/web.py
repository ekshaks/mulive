import argparse

from mulive.core.server_config import web_config
from mulive.core.pipeline_helpers import add_tts, add_text_sinks
from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    final_transcript_text,
    turn_detector,
    stt,
)


async def run_session(session, tts_mode=None, stt_provider="faster_whisper", stt_model=None, model_size="tiny"):
    """
    DSL version of the basic pipeline.

    audio -> turn detector -> STT -> text sinks
    """
    await session.wait_until_ready()
    subs = SubGroup()

    audio = Stream.source(session.audio_input, name="audio")
    turn = audio | turn_detector()
    transcripts = turn.segments | stt(
        provider=stt_provider,
        model=stt_model,
        model_size=model_size,
    )
    user_text = transcripts | final_transcript_text()

    add_text_sinks(user_text, session, role="user", subs=subs)
    add_tts(user_text, session.audio_output, turn.signals, subs=subs, mode=tts_mode, provider="piper")

    try:
        await session.closed.wait()
    finally:
        subs.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL WebRTC pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", action="store_true", help="Play Piper TTS on the server speaker.")
    group.add_argument("--tts-browser", action="store_true", help="Stream Piper TTS to the browser audio track.")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--stt-provider", default="faster_whisper")
    parser.add_argument("--stt-model")
    return parser.parse_args()


if __name__ == "__main__":
    from mulive.server.server_asyncio import Server

    args = parse_args()
    tts_mode = "local" if args.tts_local else "browser" if args.tts_browser else None

    config = web_config(use_https=not args.http, debug=False)

    server = Server(
        run_session=lambda session: run_session(
            session,
            tts_mode=tts_mode,
            stt_provider=args.stt_provider,
            stt_model=args.stt_model,
            model_size=args.model_size,
        ),
        config=config,
    )
    server.run()
