import argparse
import os
from pathlib import Path

from mulive.apps.app_config import app_section
from mulive.core.auth import AppAuthentication
from mulive.core.server_config import web_config
from mulive.core.user_profiles import load_user_directory
from mulive.core.runtime_paths import users_path
from mulive.server.server_asyncio import Server
from mulive._resources import packaged_path

from .loader import load_app_catalog


DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "muapps" / "apps.yml"


def _collect_warm_up_targets(registry):
    """Scan enabled apps' merged configs for STT / TTS pairs to preload.

    Returns ``(stt_targets, tts_providers)`` where ``stt_targets`` is a
    dedup'd list of ``(provider, model_size, stt_kwargs_tuple)`` tuples
    and ``tts_providers`` is a set of provider names. Only providers
    whose in-process warm-up meaningfully moves cost off the hot path
    are considered.
    """
    stt_targets: list[tuple] = []
    seen_stt: set[tuple] = set()
    tts_providers: set[str] = set()

    for app in registry.enabled_apps():
        stt = app_section(app.config, "stt")
        tts = app_section(app.config, "tts")

        provider = stt.get("provider")
        if provider == "faster_whisper":
            model_size = stt.get("model_size", "base")
            kwargs = stt.get("kwargs") or {}
            # dict -> stable tuple key for dedup + kwargs pass-through.
            kwargs_key = tuple(sorted(kwargs.items()))
            key = (provider, model_size, kwargs_key)
            if key not in seen_stt:
                seen_stt.add(key)
                stt_targets.append(key)

        if tts.get("enabled", True):
            tts_provider = tts.get("provider")
            if tts_provider in {"kokoro_onnx", "piper"}:
                tts_providers.add(tts_provider)

    return stt_targets, tts_providers


def _warm_up_from_registry(registry, infra):
    """Run STT / VAD / TTS warm-up hooks for the enabled apps.

    Guarded by ``mulive.warm_up: true`` in the merged infra config.
    ``silero_backend: onnx`` in the infra flips the VAD backend env var
    before any inference runs.
    """
    server = infra.get("server") or {}
    if not bool(mulive.get("warm_up", False)):
        return

    silero_backend = infra.get("silero_backend")
    if isinstance(silero_backend, str) and silero_backend:
        os.environ.setdefault("SILERO_BACKEND", silero_backend)

    stt_targets, tts_providers = _collect_warm_up_targets(registry)

    for provider, model_size, kwargs_key in stt_targets:
        from mulive.core.stt.whisper import warm_up as _warm_stt

        _warm_stt(mode=provider, model_size=model_size, **dict(kwargs_key))

    if os.environ.get("SILERO_BACKEND", "torch").lower() == "onnx":
        from mulive.core.turndet import warm_up_vad as _warm_vad

        _warm_vad()

    if "kokoro_onnx" in tts_providers:
        from mulive.core.tts_providers.kokoro_onnx import warm_up as _warm_kokoro

        _warm_kokoro()
    if "piper" in tts_providers:
        from mulive.core.tts_providers.piper import warm_up as _warm_piper

        _warm_piper()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Mulive app dashboard.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)

    tls_group = parser.add_mutually_exclusive_group()
    tls_group.add_argument("--https", action="store_true")
    tls_group.add_argument("--http", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    registry, catalog = load_app_catalog(Path(args.catalog))
    catalog_path = Path(args.catalog).resolve()
    user_directory = load_user_directory(users_path(catalog_path, catalog))
    for unavailable in registry.unavailable_apps():
        print(
            "App unavailable: "
            f"{unavailable.title} ({unavailable.bundle_path}): "
            f"{unavailable.reason}"
        )
    server_config = dict(app_section(catalog, "server"))

    use_https = (
        True
        if args.https
        else False
        if args.http
        else server_config.pop("https", True)
    )
    host = args.host or server_config.pop("host", "0.0.0.0")
    port = args.port or int(server_config.pop("port", 9000))

    # Preload heavyweight models (Whisper / Silero VAD / Kokoro-ONNX / Piper)
    # before the WebRTC listener is opened, so first-connect and first-speech
    # don't pay lazy-load latency on the user's hot path. Opt-in via
    # ``mulive.warm_up: true`` in the merged infra config.
    _warm_up_from_registry(registry, catalog)

    server = Server(
        app_registry=registry,
        user_directory=user_directory,
        config=web_config(use_https=use_https, **server_config),
        authentication=AppAuthentication.from_environment(
            login_html_path=packaged_path("client", "login.html"),
            secure_cookie=use_https,
        ),
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
