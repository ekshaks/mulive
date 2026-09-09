"""Transport-agnostic transcribe/respond/delivery coordination."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ConversationReply:
    text: str
    context: Any = None


Respond = Callable[[str], ConversationReply | Awaitable[ConversationReply]]
ReportDelivery = Callable[[ConversationReply, bool], None | Awaitable[None]]


class ConversationTurnProcessor:
    """Runs application conversation work without knowing its transport or policy."""

    def __init__(
        self,
        transcribe: Callable[[bytes], Awaitable[str]],
        respond: Respond,
        report_delivery: ReportDelivery | None = None,
        *,
        recovery_text: str | None = None,
    ) -> None:
        self._transcribe = transcribe
        self._respond = respond
        self._report_delivery = report_delivery
        self._recovery_text = recovery_text

    async def handle_turn(self, pcm16: bytes, turn, is_current: Callable[[], bool]):
        text = await self._transcribe(pcm16)
        if not is_current():
            return None
        try:
            reply = self._respond(text)
            if inspect.isawaitable(reply):
                reply = await reply
        except Exception:
            if self._recovery_text is None:
                raise
            reply = ConversationReply(self._recovery_text)
        if not is_current():
            return None
        if not isinstance(reply, ConversationReply):
            raise TypeError("respond() must return ConversationReply")
        return text, reply

    async def report_delivery(self, reply: ConversationReply, text_delivered: bool) -> None:
        if self._report_delivery is None:
            return
        result = self._report_delivery(reply, text_delivered)
        if inspect.isawaitable(result):
            await result


class WebSocketTurnHandler:
    """VoiceSession adapter for protocol events, speech, and delivery facts."""

    def __init__(self, conversation: ConversationTurnProcessor, tts_flow) -> None:
        self.conversation = conversation
        self.tts_flow = tts_flow
        self._session = None
        self._speaker = None
        self._closed = False

    def bind_session(self, session) -> None:
        self._session = session
        if self.tts_flow.enabled:
            self._speaker = self.tts_flow.create_speaker(session.audio_output)

    async def wait_ready(self) -> None:
        return None

    async def run(self, pcm16, turn, is_current, emit, _audio_output) -> None:
        completed = await self.conversation.handle_turn(pcm16, turn, is_current)
        if completed is None:
            return
        text, reply = completed
        await emit({"type": "transcript.final", "turn_id": turn.turn_id, "text": text})
        if not is_current() or self._session is None:
            return
        try:
            result = await self._session.speak(reply.text)
        except BaseException:
            await self.conversation.report_delivery(reply, False)
            raise
        await self.conversation.report_delivery(reply, result.text_delivered)
        if is_current():
            await emit({"type": "turn.finished", "turn_id": turn.turn_id, "outcome": "responded"})

    async def speak(self, text: str, cancelled, _audio_output) -> None:
        if self._speaker is None:
            raise RuntimeError("voice handler is not bound to a TTS speaker")
        await self._speaker.speak(text, cancelled)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._speaker is not None:
            close = getattr(self._speaker, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
