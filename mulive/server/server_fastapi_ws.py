import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect

from mulive.core.ws_voice_protocols import STT_LLM_TTS_Flow, VoiceHandler
from .fastapi_voice_ws_transport import local_principal, mount_voice_transport
from .._resources import packaged_path

logger = logging.getLogger("uvicorn.error")
_pipeline: STT_LLM_TTS_Flow | None = None  # compatibility hook for embedders/tests

def create_app(
    config: dict | None = None,
    voice_handler: VoiceHandler | None = None,
    run_session=None,
) -> FastAPI:
    """Create the generic voice WebSocket application.

    The transport server uses the generic STT_LLM_TTS_Flow defaults unless an
    application supplies a complete VoiceHandler.
    """
    app = FastAPI()
    pipeline = voice_handler
    frontend_dir = packaged_path("client")
    assert frontend_dir.exists(), f"Frontend directory not found: {frontend_dir}"
    app.mount("/client", StaticFiles(directory=frontend_dir), name="client")

    def voice_pipeline() -> STT_LLM_TTS_Flow:
        nonlocal pipeline
        global _pipeline
        if pipeline is not None:
            return pipeline
        if _pipeline is not None:
            return _pipeline
        if pipeline is None:
            pipeline = STT_LLM_TTS_Flow(config=config)
            _pipeline = pipeline
        return pipeline

    async def default_run_session(session):
        handler = session.pipeline or voice_pipeline()
        if session.pipeline is None:
            session.set_handler(handler)
        await session.closed.wait()

    async def prepare_default_runner():
        selected = voice_pipeline()
        if hasattr(selected, "wait_ready"):
            await selected.wait_ready()

    async def close_default_runner():
        selected = pipeline
        if selected is not None and hasattr(selected, "aclose"):
            await selected.aclose()

    default_run_session.prepare = prepare_default_runner
    default_run_session.close = close_default_runner

    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(frontend_dir / "websocket/client.html")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_text()
                await ws.send_text(f"Echo: {msg}")
        except WebSocketDisconnect:
            logger.info("echo WebSocket disconnected")
        except Exception:
            logger.exception("echo WebSocket error")
            await ws.close()

    mount_voice_transport(
        app,
        path="/voice",
        run_session=run_session or default_run_session,
        principal_resolver=local_principal,
    )

    # Test and embedding hooks, matching the aio WebRTC server's injected runner.
    app.state.get_voice_pipeline = voice_pipeline
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mulive.server.server_fastapi_ws:app", host="127.0.0.1", port=8200, reload=True, workers=1)
