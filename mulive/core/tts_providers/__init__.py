from .factory import TTSConfig, create_session_tts_provider, create_tts_provider
from .sink import PlaybackState, TTSProvider, tts_sink

_PROVIDER_EXPORTS = {
    "GeminiTTSProvider": (".gemini", "GeminiTTSProvider"),
    "KokoroFastApiTTSProvider": (".kokoro_fastapi", "KokoroFastApiTTSProvider"),
    "KokoroOnnxTTSProvider": (".kokoro_onnx", "KokoroOnnxTTSProvider"),
    "PiperTTSProvider": (".piper", "PiperTTSProvider"),
}


def __getattr__(name):
    try:
        module_name, export_name = _PROVIDER_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    return getattr(import_module(module_name, __name__), export_name)
