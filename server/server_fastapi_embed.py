"""Public FastAPI mounting API for Mulive Embed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.staticfiles import StaticFiles

from .core.embed_stt_tts import EmbedRuntime, TranscriptCallback
from .fastapi_voice_ws_transport import PrincipalResolver, local_principal, mount_voice_transport


def mount_voice(
    app,
    websocket_path: str = "/mulive-ws",
    *,
    client_path: str = "/mulive/client",
    config: dict[str, Any] | None = None,
    on_transcript: TranscriptCallback | None = None,
    principal_resolver: PrincipalResolver | None = None,
    serve_client: bool = True,
) -> EmbedRuntime:
    """Mount Mulive STT/TTS onto an existing FastAPI application.

    ``config`` uses the ordinary app-bundle ``stt:`` and ``tts:`` sections.
    Without a principal resolver, this is intentionally anonymous development
    mode; production applications must provide their own resolver.
    """
    if not websocket_path.startswith("/") or websocket_path == "/":
        raise ValueError("websocket_path must be a non-root absolute path")
    if not client_path.startswith("/") or client_path == "/":
        raise ValueError("client_path must be a non-root absolute path")
    websocket_path = websocket_path.rstrip("/")
    client_path = client_path.rstrip("/")
    runtime = EmbedRuntime(config=config, on_transcript=on_transcript)
    run_session = runtime.create_session_runner()
    run_session.prepare = runtime.wait_ready
    run_session.close = runtime.aclose
    mount_voice_transport(
        app,
        path=websocket_path,
        run_session=run_session,
        principal_resolver=principal_resolver or local_principal,
        handler_factory=lambda _principal: runtime.create_handler(),
    )
    if serve_client:
        client_dir = Path(__file__).resolve().parents[1] / "client" / "embed"
        app.mount(
            client_path,
            StaticFiles(directory=client_dir),
            name=f"mulive-{client_path.strip('/').replace('/', '-')}"
        )
    return runtime
