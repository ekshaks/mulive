"""One session's voice conversation and serialized model work.

Design: An AsyncControllerFlow owns history and processes speech, transcripts, and
model results in one queue. Barge-in stops playback while backend work continues.

Catches: Side questions get "I heard you" while the current model call runs.
Their answer waits for that call or its 30-second timeout. History has no length
limit, so a long session may exceed the model's input limit. Interrupted speech
may already be recorded as a full assistant response.

User speaks
  → turn detector emits audio segment
  → speech-to-text emits final TranscriptEvent("Where is my order?")
  → ConversationHarness.submit(event)
  → ConversationHarness.run() receives it from its queue
  → _on_transcript()
      - adds user text to history
      - creates ModelRequest(id=1, history=...)
      - marks it active
  → EffectRunner starts _run_model(ModelRequest)
  → model runner runs with the serialized history
  → _run_model returns ModelResult(id=1, text="It shipped yesterday.")
  → EffectRunner calls ConversationHarness.submit(ModelResult)
  → ConversationHarness.run() receives ModelResult
  → _on_result()
      - verifies id=1 is still active
      - adds assistant text to history
      - emits AppOutput(messages=[...])
  → conversation.assistant_text stream
  → browser transcript + text-to-speech


  _run_model raises TimeoutError
  → EffectRunner converts it to ModelFailed(id=1, reason="TimeoutError")
  → ConversationHarness receives ModelFailed
  → emits:
      assistant text: "I couldn't answer that. Please try again."
      client event: {name: "web_answer", result: "error", ...}


second TranscriptEvent
  → ConversationHarness stores it in history
  → active request exists
  → emits "I heard you."
  → request 1 completes or times out
  → ConversationHarness starts request 2 with both user messages in history
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any, Literal

from mulive.apps.app_output import output
from mulive.apps.effects import EffectRunner
from mulive.apps.events import FeedbackEvent
from mulive.apps.model_runner import (
    ModelRunner,
    ModelRunnerUnavailableError,
    create_model_runner,
)
from mulive.core.async_controller_flow import AsyncControllerFlow
from mulive.core.events import TranscriptEvent
from mulive.core.logging_utils import monitor_time
from mulive.core.stream_dsl import SubGroup, map_filter_items
from mulive.core.turn import SpeechStarted

DEFAULT_GROQ_MODEL = "groq:openai/gpt-oss-20b"
DEFAULT_LLM_TIMEOUT_S = 30


@dataclass(frozen=True)
class ConversationEntry:
    role: Literal["user", "assistant", "background"]
    text: str


@dataclass(frozen=True)
class ModelRequest:
    request_id: int
    user_turn: int
    history: tuple[ConversationEntry, ...]
    started_at: float

    def prompt(self) -> str:
        """Give a stateless text model the conversation visible to this session."""
        history = [
            {"role": item.role, "text": item.text}
            for item in self.history
        ]
        return (
            "Continue this voice conversation. Answer the latest user message. "
            "A background entry is an earlier model result that was not spoken; "
            "use it only if it still applies. The user may have interrupted "
            "earlier assistant speech and may not have heard it in full.\n"
            + json.dumps(history, ensure_ascii=False)
        )


@dataclass(frozen=True)
class ModelResult:
    request_id: int
    text: str


@dataclass(frozen=True)
class ModelFailed:
    request_id: int
    reason: str


class ConversationHarness(AsyncControllerFlow):
    """Own conversation state and model work; expose text and client events."""

    def __init__(
        self,
        *,
        system_prompt: str,
        llm_model=DEFAULT_GROQ_MODEL,
        llm_timeout_s=DEFAULT_LLM_TIMEOUT_S,
        tools: Sequence[Callable[..., Any]] = (),
        runner_factory: Callable[..., ModelRunner] | None = None,
    ):
        super().__init__(self, name="web_conversation_harness")
        self.history: list[ConversationEntry] = []
        self._user_turn = 0
        self._request_id = 0
        self._active: ModelRequest | None = None
        self._queued_started_at: float | None = None
        self._hearing = False
        self._unclear = False
        factory = runner_factory or create_model_runner
        try:
            self._model_runner = (
                factory(
                    llm_model,
                    instructions=system_prompt,
                    tools=tools,
                )
                if llm_model is not None else None
            )
        except ModelRunnerUnavailableError as exc:
            warnings.warn(f"LLM control disabled: {exc}", RuntimeWarning, stacklevel=2)
            self._model_runner = None
        self._acknowledge = self._model_runner is not None
        self._llm_timeout_s = llm_timeout_s
        self._subs = SubGroup()
        self._effects = EffectRunner(self.submit, name="web_answer")
        self._effects.register(
            ModelRequest,
            self._run_model,
            on_error=lambda request, exc: ModelFailed(request.request_id, type(exc).__name__),
        )
        self.assistant_text = self.outputs | map_filter_items(
            map_fn=lambda item: " ".join(item.messages).strip(),
            filter_fn=bool,
            name="web_conversation_harness_messages",
        )
        # Browser-only events; spoken messages are in assistant_text.
        self.client_events = self.outputs | map_filter_items(
            map_fn=lambda item: item.feedback,
            filter_fn=lambda item: item is not None,
            name="web_conversation_harness_client_events",
        )

    def connect(self, final_transcripts, speech_signals) -> None:
        """Feed final STT events and speech starts into the conversation queue."""
        final_transcripts.to(self.input_sink(), name="web_transcripts_to_conversation", subs=self._subs)
        speech_signals.to(self.input_sink(), name="web_speech_to_conversation", subs=self._subs)

    async def _run_model(self, request: ModelRequest) -> ModelResult:
        if self._model_runner is None:
            reply = next(item.text for item in reversed(request.history) if item.role == "user")
        else:
            try:
                reply = await asyncio.wait_for(
                    self._model_runner.run(request.prompt()),
                    timeout=self._llm_timeout_s,
                )
            except ModelRunnerUnavailableError as exc:
                warnings.warn(f"LLM control disabled: {exc}", RuntimeWarning, stacklevel=2)
                self._model_runner = None
                raise
        return ModelResult(request.request_id, reply)

    async def aclose(self) -> None:
        await self._effects.close()
        self.close()
        await self._subs.aclose()

    async def run(self, ctx) -> None:
        # Consume every event. wait_for(ModelResult) would discard interruptions.
        while True:
            event = await ctx.next_event()
            if isinstance(event, SpeechStarted):
                self._hearing = True
            elif isinstance(event, TranscriptEvent) and event.is_final:
                self._on_transcript(ctx, event.text.strip())
            elif isinstance(event, ModelResult):
                self._on_result(ctx, event)
            elif isinstance(event, ModelFailed):
                self._on_failure(ctx, event)

    def _on_transcript(self, ctx, text: str) -> None:
        self._hearing = False
        if not text:
            self._unclear = True
            ctx.emit(output(None, "I didn't catch that. Please say it again."))
            return

        self._unclear = False
        started_at = time.perf_counter()
        self._user_turn += 1
        self.history.append(ConversationEntry("user", text))
        if self._active is None:
            self._start_request(started_at)
        else:
            if self._queued_started_at is None:
                self._queued_started_at = started_at
            if self._acknowledge:
                ctx.emit(output(None, "I heard you."))

    def _start_request(self, started_at: float | None = None) -> None:
        self._request_id += 1
        request = ModelRequest(
            self._request_id,
            self._user_turn,
            tuple(self.history),
            time.perf_counter() if started_at is None else started_at,
        )
        self._queued_started_at = None
        self._active = request
        self._effects.start(request)

    def _on_result(self, ctx, event: ModelResult) -> None:
        request = self._active
        if request is None or event.request_id != request.request_id:
            return
        self._active = None
        monitor_time(
            "conversation",
            "model_result",
            time.perf_counter() - request.started_at,
            request_id=request.request_id,
            user_turn=request.user_turn,
        )
        text = (event.text or "").strip()
        if not text:
            self._on_current_failure(ctx, "empty response", request)
            return

        if self._hearing or self._unclear or self._user_turn != request.user_turn:
            self.history.append(ConversationEntry("background", text))
            if not self._hearing and not self._unclear and self._user_turn != request.user_turn:
                self._start_request(self._queued_started_at)
            return

        self.history.append(ConversationEntry("assistant", text))
        ctx.emit(output(None, text))

    def _on_failure(self, ctx, event: ModelFailed) -> None:
        request = self._active
        if request is None or event.request_id != request.request_id:
            return
        self._active = None
        monitor_time(
            "conversation",
            "model_failed",
            time.perf_counter() - request.started_at,
            request_id=request.request_id,
            user_turn=request.user_turn,
            reason=event.reason,
        )
        self._on_current_failure(ctx, event.reason, request)

    def _on_current_failure(self, ctx, reason: str, request: ModelRequest) -> None:
        if self._user_turn != request.user_turn:
            if not self._hearing and not self._unclear:
                self._start_request(self._queued_started_at)
            return
        if self._hearing or self._unclear:
            return
        message = "I couldn't answer that. Please try again."
        self.history.append(ConversationEntry("assistant", message))
        ctx.emit(
            output(
                None,
                message,
                feedback=FeedbackEvent("web_answer", "error", {"reason": reason}),
            )
        )
