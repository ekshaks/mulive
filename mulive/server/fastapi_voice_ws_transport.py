"""Shared FastAPI WebSocket transport for Mulive voice sessions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from mulive.core.ws_voice_protocols import VoicePrincipal, VoiceSession

logger = logging.getLogger("uvicorn.error")
PrincipalResolver = Callable[
    [WebSocket], VoicePrincipal | None | Awaitable[VoicePrincipal | None]
]


def local_principal(_websocket: WebSocket) -> VoicePrincipal:
    return VoicePrincipal("local", "Local")


async def _safe_principal(resolver: PrincipalResolver, websocket: WebSocket):
    principal = resolver(websocket)
    if inspect.isawaitable(principal):
        principal = await principal
    return principal


def mount_voice_transport(
    app: FastAPI,
    *,
    path: str,
    run_session: Callable[[VoiceSession], Awaitable[Any]],
    principal_resolver: PrincipalResolver,
    handler_factory: Callable[[VoicePrincipal], Any] | None = None,
) -> FastAPI:
    """Mount the protocol loop and application/session lifecycle hooks."""
    if not path.startswith("/") or path == "/":
        raise ValueError("voice WebSocket path must be a non-root absolute path")

    @app.on_event("startup")
    async def prepare_voice_runner():
        prepare = getattr(run_session, "prepare", None)
        if prepare is not None:
            await prepare()

    @app.on_event("shutdown")
    async def close_voice_runner():
        close = getattr(run_session, "close", None)
        if close is not None:
            await close()

    @app.websocket(path)
    async def voice_endpoint(websocket: WebSocket):
        principal = await _safe_principal(principal_resolver, websocket)
        if principal is None:
            await websocket.close(code=4401, reason="voice authentication required")
            return
        if not isinstance(principal, VoicePrincipal):
            await websocket.close(code=1008, reason="invalid voice principal")
            return
        await websocket.accept()
        logger.info("voice socket accepted client=%s subject=%s", websocket.client, principal.subject)
        # Bind before the receive loop starts so hello followed immediately by
        # PCM cannot commit a turn against an unbound handler.
        handler = None
        try:
            handler = handler_factory(principal) if handler_factory is not None else None
            session = VoiceSession(principal, handler, websocket.send_json, websocket.send_bytes)
        except Exception:
            if handler is not None:
                close = getattr(handler, "aclose", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            await websocket.close(code=1011, reason="voice handler unavailable")
            return
        runner_task = asyncio.create_task(run_session(session), name="voice-session-runner")
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    event = json.loads(message["text"])
                    await session.handle_json(event)
                elif message.get("bytes") is not None:
                    await session.handle_pcm16(message["bytes"])
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("voice protocol error: %s", exc)
            try:
                await websocket.send_json({"type": "error", "text": "invalid voice protocol event"})
            except Exception:
                logger.debug("unable to send protocol error", exc_info=True)
        except WebSocketDisconnect:
            pass
        finally:
            await session.cancel()
            if not runner_task.done():
                runner_task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)

    return app
