import argparse
import os

from mulive.core.server_config import web_config
from mulive._resources import packaged_path
from mulive.apps.prompts import load_system_prompt
from mulive.core.stt import STTConfig
from mulive.core.pipeline_helpers import non_empty_text, to_user
from mulive.core.tts_providers import TTSConfig, create_session_tts_provider
from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    client_message_sink,
    filter_items,
    turn_detector,
    stt,
)
from mulive.apps.conversation_harness import (
    ConversationHarness,
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_TIMEOUT_S,
)
async def today_date() -> str:
    """Return the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

async def lookup_order(order_id: str) -> dict:
    """Return the status for one order without changing order data."""
    return {
        "order_id": order_id,
        "status": "unavailable",
        "reason": "Order data source is not configured.",
    }


TOOLS = (lookup_order, today_date)
VOICE_SYSTEM_PROMPT = load_system_prompt(
    packaged_path("quickstart", "prompts.yml"), "web_voice"
)


async def run_session(
    session,
    tts_config: TTSConfig | None = None,
    stt_config: STTConfig = STTConfig(variant="small"),
    llm_model=DEFAULT_GROQ_MODEL,
    llm_timeout_s=DEFAULT_LLM_TIMEOUT_S,
):
    await session.wait_until_ready()
    subs = SubGroup()
    tts_provider = create_session_tts_provider(session, subs, tts_config)
    conversation = None

    try:
        if not os.environ.get("GROQ_API_KEY"):
            llm_model = None

        audio = Stream.source(session.audio_input, name="audio")
        turn = audio | turn_detector()
        transcripts = turn.value | stt(stt_config)
        final_transcripts = transcripts | filter_items(lambda event: event.is_final)
        user_text = final_transcripts | non_empty_text(name="web_user_text")
        conversation = ConversationHarness(
            system_prompt=VOICE_SYSTEM_PROMPT,
            llm_model=llm_model,
            llm_timeout_s=llm_timeout_s,
            tools=TOOLS,
        )
        conversation.connect(final_transcripts, turn.started)

        user_text | to_user(session=session, role="user", subs=subs)
        conversation.assistant_text | to_user(
            session=session, role="assistant", subs=subs,
            tts=tts_config, tts_provider=tts_provider, interrupts=turn.started,
        )
        conversation.client_events.to(
            client_message_sink(session),
            name="web_conversation_harness_client_events",
            subs=subs,
        )
        conversation.start()
        await session.closed.wait()
    finally:
        if conversation is not None:
            await conversation.aclose()
        await subs.aclose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL WebRTC pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", dest="tts_mode", action="store_const", const="local", help="Play Piper TTS on the server speaker.")
    group.add_argument("--tts-browser", dest="tts_mode", action="store_const", const="browser", help="Stream Piper TTS to the browser audio track.")
    parser.set_defaults(tts_mode="browser")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    parser.add_argument("--model-variant", default="small")
    parser.add_argument("--stt-provider", default="faster_whisper")
    parser.add_argument("--stt-model")
    parser.add_argument("--llm-model", default=DEFAULT_GROQ_MODEL, help="Groq model used when GROQ_API_KEY is set.")
    parser.add_argument("--llm-timeout-s", type=float, default=DEFAULT_LLM_TIMEOUT_S)
    return parser.parse_args()


if __name__ == "__main__":
    from mulive.server.server_asyncio import Server

    args = parse_args()

    config = web_config(use_https=not args.http, debug=False)
    tts_config = TTSConfig(provider="piper", mode=args.tts_mode)
    stt_config = STTConfig(
        provider=args.stt_provider,
        model=args.stt_model,
        variant=args.model_variant,
    )
    if stt_config.provider == "faster_whisper":
        from mulive.core.stt.whisper import warm_up
        warm_up(stt_config)

    server = Server(
        run_session=lambda session: run_session(
            session,
            tts_config=tts_config,
            stt_config=stt_config,
            llm_model=args.llm_model,
            llm_timeout_s=args.llm_timeout_s,
        ),
        config=config,
    )
    server.run()
