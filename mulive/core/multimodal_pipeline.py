import asyncio
import uuid

from .events import ClientTranscriptMessage
from .turn_source import SpeechStarted, VoiceTurn
from .voice_engine import run_voice_turn


class WebRTCVoiceTurnRunner:
    """Bridge completed Rx transcripts into the shared voice-turn engine.

    VAD and STT deliberately stay in the existing Rx pipeline.  This runner
    owns only the post-transcript turn lifetime, including VAD barge-in.
    """

    def __init__(self, *, session, answer, speak):
        self.session = session
        self.answer = answer
        self.speak = speak
        self._generation = 0
        self._input_turn: VoiceTurn | None = None
        self._active_turn: VoiceTurn | None = None
        self._task: asyncio.Task | None = None

    def on_vad_event(self, event) -> None:
        if event.speech_started is not None:
            self._cancel_active()
            if self._input_turn is not None:
                self._input_turn.cancelled.set()
            self._input_turn = event.speech_started.turn

    def on_speech_started(self, event: SpeechStarted) -> None:
        """Cancel an active response when a new user turn begins."""
        self._cancel_active()

    def start(self, text: str, turn: VoiceTurn | None = None) -> None:
        if turn is not None and turn.cancelled.is_set():
            return
        self._cancel_active()
        self._generation += 1
        generation = self._generation
        turn = turn.with_generation(generation) if turn is not None else VoiceTurn(uuid.uuid4().hex, generation=generation)
        self._input_turn = None
        self._active_turn = turn
        self._task = asyncio.create_task(
            self._run(text, turn), name=f"webrtc-voice:{turn.turn_id}"
        )

    def cancel(self) -> asyncio.Task | None:
        task = self._cancel_active()
        if self._input_turn is not None:
            self._input_turn.cancelled.set()
            self._input_turn = None
        return task

    def _cancel_active(self) -> asyncio.Task | None:
        self._generation += 1
        if self._active_turn is not None:
            self._active_turn.cancelled.set()
            self._active_turn = None
        task = self._task
        if self._task is not None:
            self._task.cancel()
            self._task = None
        clear = getattr(self.session.audio_output, "clear", None)
        if clear is not None:
            clear()
        return task

    async def aclose(self) -> None:
        task = self.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self, text: str, turn: VoiceTurn) -> None:
        def is_current() -> bool:
            return self._active_turn is turn and not turn.cancelled.is_set()

        async def emit(event: dict) -> None:
            if not is_current():
                return
            if event["type"] == "transcript.final":
                self.session.send_to_client(ClientTranscriptMessage(role="user", content=event["text"]))
            elif event["type"] == "response.text":
                self.session.send_to_client(ClientTranscriptMessage(role="assistant", content=event["text"]))
            else:
                self.session.send_to_client(event)

        try:
            await run_voice_turn(
                b"",
                turn=turn,
                transcribe_turn=None,
                stt_timeout_seconds=0,
                llm_model="",
                is_current=is_current,
                emit=emit,
                audio_output=self.session.audio_output,
                response_id_factory=lambda: uuid.uuid4().hex,
                get_transcript=lambda: _completed(text),
                answer=self.answer,
                speak=self.speak,
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if is_current():
                self.session.send_to_client(
                    {
                        "type": "error",
                        "turn_id": turn.turn_id,
                        "text": f"pipeline failed: {type(exc).__name__}",
                    }
                )
        finally:
            if self._active_turn is turn:
                self._active_turn = None
                self._task = None


async def _completed(value):
    return value
