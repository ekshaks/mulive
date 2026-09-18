import os
import threading
import time

from ..logging_utils import monitor_time

DEBUG_MLX_STT = os.getenv("DEBUG_MLX_STT") == "1"


def _log(message):
    if DEBUG_MLX_STT:
        print(message)


def load_mlx_model(model_id):
    import mlx.core as mx
    from mlx_whisper import load_models as mlx_load_models

    _log(f"[mlx-stt] Loading mlx model {model_id} on thread={threading.get_ident()}")
    start = time.perf_counter()
    dtype = mx.float16
    model = mlx_load_models.load_model(path_or_hf_repo=model_id, dtype=dtype)
    if DEBUG_MLX_STT:
        monitor_time("stt", "load_model", time.perf_counter() - start, provider="mlx", model=model_id)
    return model


def get_mlx_whisper_model(model_size: str = "base", model_id=None):
    if model_size == "turbo":
        model_id = "mlx-community/whisper-large-v3-turbo"
    model_id = model_id or f"mlx-community/whisper-{model_size}-mlx"
    return load_mlx_model(model_id)


def mlx_feats(audio_data, n_mels: int = 80):
    import mlx.core as mx
    from mlx_whisper import audio as mlx_audio

    array = mx.array(audio_data)
    data = mlx_audio.pad_or_trim(array)
    mels = mlx_audio.log_mel_spectrogram(data, n_mels)
    mx.eval(mels)
    return mels


def infer_mlx(audio_data, model):
    import mlx.core as mx
    from mlx_whisper import decoding as mlx_decoding

    _log(f"[mlx-stt] infer_mlx start audio_shape={getattr(audio_data, 'shape', None)} thread={threading.get_ident()}")
    start = time.perf_counter()
    mels = mlx_feats(audio_data, model.dims.n_mels)[None].astype(mx.float16)
    if DEBUG_MLX_STT:
        monitor_time("stt", "features", time.perf_counter() - start, provider="mlx", shape=mels.shape)
    options = mlx_decoding.DecodingOptions(language="en")
    result = mlx_decoding.decode(model, mels, options=options)
    if DEBUG_MLX_STT:
        monitor_time("stt", "decode", time.perf_counter() - start, provider="mlx")
    return result[0].text
