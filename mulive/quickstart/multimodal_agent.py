import argparse
import os
from pathlib import Path

from mulive.core.llm_utils import call_llm, create_agent
from mulive.core.multimodal_pipeline import WebRTCVoiceTurnRunner
from mulive.core.pipeline_helpers import make_tts_provider
from mulive.core.stream_dsl import Stream, SubGroup, final_transcripts, stt, turn_detector
from mulive.core.server_config import web_config
from mulive._resources import packaged_path

PROMPTS_FILE = packaged_path("quickstart", "prompts.yml")


async def run_multimodal_session(
    session,
    *,
    mode="av",
    stt_provider="faster_whisper",
    stt_model_size="tiny",
    stt_model=None,
    stt_language="en",
    stt_kwargs=None,
    llm_model,
    prompts_path=PROMPTS_FILE,
    prompt_id="visual_solver",
    agent_name="Agent",
    tts_mode=None,
    tts_provider="piper",
):
    """Compose the generic voice core for the visual quickstart demo."""
    await session.wait_until_ready()
    subs = SubGroup()
    latest_frame = Stream.source(session.video_input, name="video").latest(
        name="latest_frame", subs=subs
    )
    turn = Stream.source(session.audio_input, name="audio") | turn_detector()
    transcripts = turn.turns | stt(
        provider=stt_provider,
        model=stt_model,
        model_size=stt_model_size,
        language=stt_language,
        **(stt_kwargs or {}),
    )
    completed = transcripts | final_transcripts()
    agent = create_agent(
        llm_model,
        prompts_path=prompts_path,
        prompt_id=prompt_id,
        name=agent_name,
    )

    async def answer(text):
        return await call_llm(agent, text, latest_frame.get(), mode)

    tts = None
    if tts_mode is not None:
        output_mode = {"local": "local", "browser": "webrtc"}.get(tts_mode)
        if output_mode is None:
            raise ValueError(f"Unknown TTS mode: {tts_mode}")
        tts = make_tts_provider(
            tts_provider,
            output_mode,
            session.audio_output if output_mode == "webrtc" else None,
        )

    async def speak(text, cancelled, _audio_output):
        if tts is not None and text.strip():
            await tts.speak(text, cancelled)

    runner = WebRTCVoiceTurnRunner(session=session, answer=answer, speak=speak)
    turn.events.to(
        lambda stream: stream.subscribe(runner.on_vad_event),
        name="quickstart_barge_in",
        subs=subs,
    )
    completed.to(
        lambda stream: stream.subscribe(
            lambda event: runner.start(event.text, event.context)
        ),
        name="quickstart_voice_turns",
        subs=subs,
    )
    try:
        await session.closed.wait()
    finally:
        latest_frame.dispose()
        subs.dispose()
        await runner.aclose()
        close = getattr(tts, "aclose", None)
        if close is not None:
            await close()


async def run_session(
    session,
    mode="av",
    tts_mode=None,
    tts_provider="kokoro_fastapi",
    stt_model_size="tiny",
    stt_provider="mlx",
    stt_model=None,
    stt_language="en",
    stt_kwargs=None,
    llm_model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
):
    await run_multimodal_session(
        session,
        mode=mode,
        stt_provider=stt_provider,
        stt_model_size=stt_model_size,
        stt_model=stt_model,
        stt_language=stt_language,
        stt_kwargs=stt_kwargs,
        llm_model=llm_model,
        prompts_path=PROMPTS_FILE,
        prompt_id="visual_solver",
        agent_name="Math Helper",
        tts_mode=tts_mode,
        tts_provider=tts_provider,
    )


def _coerce(value: str):
    """Turn a CLI ``--stt-kwarg KEY=VALUE`` string into an int/float/bool/str."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_kv_list(pairs):
    """Parse a list of ``key=value`` strings into a dict, coercing values."""
    result = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--stt-kwarg expects key=value, got: {pair!r}")
        key, _, value = pair.partition("=")
        result[key.strip()] = _coerce(value)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multimodal DSL WebRTC pipeline.")
    parser.add_argument("--mode", choices=["a", "av"], default="av")
    parser.add_argument("--stt-model-size", default="tiny")
    parser.add_argument("--stt-provider", default="faster_whisper",
                        help="STT provider: mlx | faster_whisper | deepgram")
    parser.add_argument("--stt-model")
    parser.add_argument("--stt-language", default="en")
    parser.add_argument(
        "--stt-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword argument forwarded to the STT backend. Repeatable. "
            "Examples: cpu_threads=1 compute_type=int8 num_workers=1 device=cpu."
        ),
    )
    parser.add_argument("--llm-model", default="groq:meta-llama/llama-4-scout-17b-16e-instruct")
    parser.add_argument(
        "--max-concurrent-sessions",
        type=int,
        default=2,
        help="Reject WebRTC offers beyond this many active sessions (0 = unlimited).",
    )
    parser.add_argument(
        "--tts-provider",
        choices=["kokoro_fastapi", "kokoro_onnx", "piper"],
        default="piper",
        help=(
            "piper (default) runs local streaming TTS; kokoro_fastapi talks to "
            "an external Kokoro-FastAPI HTTP server; kokoro_onnx runs an ONNX "
            "model in-process."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts-local", action="store_true", help="Play TTS on the server speaker.")
    group.add_argument("--tts-browser", action="store_true", help="Stream TTS to the browser audio track.")
    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true", default=True, help="Serve HTTPS with local certs. Default.")
    tls_group.add_argument("--http", action="store_true", help="Serve plain HTTP.")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help=(
            "Disable debug logging and per-chunk audio metrics. "
            "Recommended on small servers: enables the cheap numpy-only "
            "active-speaker check."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    from mulive.server.server_asyncio import Server

    args = parse_args()
    tts_mode = "local" if args.tts_local else "browser" if args.tts_browser else None

    config = web_config(
        use_https=not args.http,
        debug=not args.no_debug,
        rms_thresh=0.025,
        input_video_sample_interval=100,
        filter_gender=None,
        max_concurrent_sessions=args.max_concurrent_sessions or None,
    )

    stt_kwargs = _parse_kv_list(args.stt_kwarg)

    # Warm up every heavyweight model BEFORE accepting connections, but only
    # for the small-server (AWS) providers where lazy first-use blocks the
    # user's hot path:
    #
    #   * faster-whisper — per-session load is 1-3 s on 1 vCPU (visible as
    #     "browser hangs on connect"). MLX on desktop stays lazy: it was
    #     already lazily loaded per WhisperSTT and is fast on Apple Silicon.
    #   * SILERO_BACKEND=onnx — first inference may download the ONNX model
    #     (visible as "mic hangs on first speech").
    #   * kokoro_onnx and Piper may need a first-use model load; warm them
    #     before accepting sessions when selected.
    if args.stt_provider == "faster_whisper":
        from mulive.core.stt.whisper import warm_up as _warm_stt
        _warm_stt(mode=args.stt_provider, model_size=args.stt_model_size, **stt_kwargs)
    if os.environ.get("SILERO_BACKEND", "onnx").lower() == "onnx":
        from mulive.core.turndet import warm_up_vad as _warm_vad
        _warm_vad()
    if args.tts_provider == "kokoro_onnx" and tts_mode is not None:
        from mulive.core.tts_providers.kokoro_onnx import warm_up as _warm_tts
        _warm_tts()
    if args.tts_provider == "piper" and tts_mode is not None:
        from mulive.core.tts_providers.piper import warm_up as _warm_piper
        _warm_piper()

    server = Server(
        run_session=lambda session: run_session(
            session,
            mode=args.mode,
            tts_mode=tts_mode,
            tts_provider=args.tts_provider,
            stt_model_size=args.stt_model_size,
            stt_provider=args.stt_provider,
            stt_model=args.stt_model,
            stt_language=args.stt_language,
            stt_kwargs=stt_kwargs,
            llm_model=args.llm_model,
        ),
        config=config,
    )
    server.run()
