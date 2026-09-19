"""Transport-neutral shared voice-turn execution."""

import asyncio

import numpy as np

from .audio_output import AudioChunk
from .llm_utils import call_groq_chat, split_spoken_written
from .turn import TurnContext


async def run_voice_turn(
    pcm16: bytes,
    *,
    context: TurnContext,
    transcribe_turn,
    stt_timeout_seconds: float,
    llm_model: str,
    is_current,
    emit,
    audio_output,
    response_id_factory,
    tts_client=None,
    get_transcript=None,
    answer=None,
    speak=None,
    split_response=split_spoken_written,
) -> None:
    """Run the shared local voice turn, rejecting output from stale generations."""
    cancelled = context.cancelled
    turn_id = context.id
    try:
        text = await (get_transcript() if get_transcript is not None else asyncio.wait_for(transcribe_turn(np.frombuffer(pcm16, dtype=np.int16)), timeout=stt_timeout_seconds))
    except TimeoutError:
        if is_current():
            await emit({"type": "error", "turn_id": turn_id, "text": "Speech recognition timed out. Try again."})
            await emit({"type": "turn.finished", "turn_id": turn_id, "outcome": "failed", "reason": "stt_timeout"})
        return
    if cancelled.is_set() or not is_current():
        return
    await emit({"type": "transcript.final", "turn_id": turn_id, "text": text})
    reply = await (answer(text) if answer is not None else call_groq_chat(
        llm_model,
        system_prompt="You are a concise voice assistant. Answer the user directly. Distinguish hypotheses from verified facts.",
        user_prompt=text,
    ))
    if reply is None or cancelled.is_set() or not is_current():
        return
    response_id = response_id_factory()
    await emit({"type": "response.started", "turn_id": turn_id, "response_id": response_id, "sample_rate": 24_000})
    parts = split_response(reply)
    spoken = parts["spoken"]
    written = parts["written"]
    await emit({"type": "response.text", "turn_id": turn_id, "response_id": response_id, "text": written})

    async def write_audio(block, sample_rate) -> None:
        if cancelled.is_set() or not is_current():
            return
        await audio_output.write(AudioChunk(block.astype(np.int16, copy=False), sample_rate))

    if speak is not None:
        await speak(spoken, cancelled, audio_output)
    else:
        from .tts_providers.kokoro_fastapi import _tts_kokoro_stream_chunks

        await _tts_kokoro_stream_chunks(spoken, cancelled, write_audio, client=tts_client)
    if not cancelled.is_set() and is_current():
        await audio_output.wait_until_drained()
        if is_current():
            await emit({"type": "response.finished", "turn_id": turn_id, "response_id": response_id})
            await emit({"type": "turn.finished", "turn_id": turn_id, "outcome": "answered"})
