"""Serve a minimal host application with the default Mulive Embed routes."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mulive import mount_voice
from mulive.apps.app_config import load_app_config


APP_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    # This is the host application's ordinary FastAPI instance.
    app = FastAPI(title="Mulive Embed Quickstart")

    # Mount Mulive's voice socket and browser SDK using standard STT/TTS config.
    config = load_app_config(APP_DIR / "config.yml")
    mount_voice(app, config=config)

    # Serve this quickstart's page assets separately from the Mulive SDK.
    app.mount("/app", StaticFiles(directory=APP_DIR), name="embed-app")

    # Open the demo page at the application root.
    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(APP_DIR / "index.html")

    return app


# ASGI entry point used by `uvicorn quickstart.embed.app:app`.
app = create_app()
