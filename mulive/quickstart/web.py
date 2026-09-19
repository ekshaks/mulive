import argparse
import os

from mulive.core.server_config import web_config
from mulive.core.pipeline_helpers import non_empty_text, to_user
from mulive.core.tts_providers import TTSConfig, create_session_tts_provider
from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    client_message_sink,
    filter_items,
    map_items,
    turn_detector,
    stt,
)
from mulive.apps.agent import Agent, DEFAULT_GROQ_MODEL, DEFAULT_LLM_TIMEOUT_S


async def run_session(
    session,
    tts_config: TTSConfig | None = None,
    stt_provider="faster_whisper",
    stt_model=None,
    model_size="small",
    llm_model=DEFAULT_GROQ_MODEL,
    llm_timeout_s=DEFAULT_LLM_TIMEOUT_S,
):
    await session.wait_until_ready()
    subs = SubGroup()
    tts_provider = create_session_tts_provider(session, subs, tts_config)
    agent = None

    try:
        if not os.environ.get("GROQ_API_KEY"):
            llm_model = None

        audio = Stream.source(session.audio_input, name="audio")
        turn = audio | turn_detector()
        transcripts = turn.value | stt(
            provider=stt_provider,
            model=stt_model,
            model_size=model_size,
        )
        final_transcripts = transcripts | filter_items(lambda event: event.is_final)
        user_text = (
            final_transcripts
            | map_items(lambda event: event.text, name="web_user_text")
            | non_empty_text()
        )
        agent = Agent(llm_model=llm_model, llm_timeout_s=llm_timeout_s)
        agent.connect(final_transcripts, turn.started)

        user_text | to_user(session=session, role="user", subs=subs)
        agent.assistant_text | to_user(
            session=session, role="assistant", subs=subs,
            tts=tts_config, tts_provider=tts_provider, interrupts=turn.started,
        )
        agent.client_events.to(
            client_message_sink(session),
            name="web_agent_client_events",
            subs=subs,
        )
        agent.start()
        await session.closed.wait()
    finally:
        if agent is not None:
            await agent.aclose()
        await subs.aclose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL WebRTC pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", dest="tts_mode", action="store_const", const="local", help="Play Piper TTS on the server speaker.")
    group.add_argument("--tts-browser", dest="tts_mode", action="store_const", const="browser", help="Stream Piper TTS to the browser audio track.")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--stt-provider", default="faster_whisper")
    parser.add_argument("--stt-model")
    parser.add_argument("--llm-model", default=DEFAULT_GROQ_MODEL, help="Groq model used when GROQ_API_KEY is set.")
    parser.add_argument("--llm-timeout-s", type=float, default=DEFAULT_LLM_TIMEOUT_S)
    return parser.parse_args()


if __name__ == "__main__":
    from mulive.server.server_asyncio import Server

    args = parse_args()

    config = web_config(use_https=not args.http, debug=False)
    tts_config = TTSConfig(provider="piper", mode=args.tts_mode) if args.tts_mode else None

    server = Server(
        run_session=lambda session: run_session(
            session,
            tts_config=tts_config,
            stt_provider=args.stt_provider,
            stt_model=args.stt_model,
            model_size=args.model_size,
            llm_model=args.llm_model,
            llm_timeout_s=args.llm_timeout_s,
        ),
        config=config,
    )
    server.run()
