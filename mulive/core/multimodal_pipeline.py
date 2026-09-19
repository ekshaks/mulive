import asyncio
import uuid

from .events import ClientTranscriptMessage
from .turn import SpeechStarted, TurnContext, VoiceTurn
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
        self._latest_started_id: str | None = None
        self._active_context: TurnContext | None = None
        self._task: asyncio.Task | None = None

    def on_speech_started(self, event: SpeechStarted) -> None:
        """Cancel an active response when a new user turn begins."""
        self._latest_started_id = event.turn_id
        self._cancel_active()

    def start(self, text: str, turn: VoiceTurn | None = None) -> None:
        """Start response work for a current completed voice turn."""
        if (
            turn is not None
            and self._latest_started_id is not None
            and turn.id != self._latest_started_id
        ):
            return
        self._cancel_active()
        self._generation += 1
        context = TurnContext(
            id=turn.id if turn is not None else uuid.uuid4().hex,
            generation=self._generation,
        )
        self._active_context = context
        self._task = asyncio.create_task(
            self._run(text, context), name=f"webrtc-voice:{context.id}"
        )

    def cancel(self) -> asyncio.Task | None:
        """Cancel active response work and return its task, if any."""
        task = self._cancel_active()
        self._latest_started_id = None
        return task

    def _cancel_active(self) -> asyncio.Task | None:
        self._generation += 1
        if self._active_context is not None:
            self._active_context.cancelled.set()
            self._active_context = None
        task = self._task
        if self._task is not None:
            self._task.cancel()
            self._task = None
        clear = getattr(self.session.audio_output, "clear", None)
        if clear is not None:
            clear()
        return task

    async def aclose(self) -> None:
        """Cancel active work and wait for its task to finish."""
        task = self.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self, text: str, context: TurnContext) -> None:
        def is_current() -> bool:
            return self._active_context is context and not context.cancelled.is_set()

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
                context=context,
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
                        "turn_id": context.id,
                        "text": f"pipeline failed: {type(exc).__name__}",
                    }
                )
        finally:
            if self._active_context is context:
                self._active_context = None
                self._task = None


async def _completed(value):
    return value
