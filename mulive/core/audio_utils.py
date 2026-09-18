import numpy as np
import av


def convert_and_resample_frame(frame: av.audio.frame.AudioFrame, target_sample_rate: int = 16000, target_channels: int = 1, debug: bool = False) -> np.ndarray:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Convert stereo 48kHz frame to mono 16kHz and return as NumPy array in (1, N) format."""
    samples = frame.to_ndarray()
    if debug: print(f"Input samples shape: {samples.shape}")
    if target_channels != 1:
        raise ValueError("only mono output is supported")
    channels = len(frame.layout.channels)
    if samples.ndim == 2 and samples.shape[0] == channels:
        channel_samples = samples
    elif samples.ndim == 2 and samples.shape[0] == 1:
        channel_samples = samples.reshape(-1, channels).T
    else:
        raise ValueError(f"unsupported audio frame shape: {samples.shape}")
    samples = np.rint(channel_samples.mean(axis=0)).astype(np.int16)
    if debug and channels > 1: print("After mono conversion: 1 channel")
    samples = resample_pcm16_mono(samples, frame.rate, target_sample_rate)
    if debug and frame.rate != target_sample_rate: print(f"After resampling, frame rate: {target_sample_rate}")
    if debug: print(f"After processing, samples shape: {samples.shape}")
    
    # Reshape to (1, N) format
    #samples = samples.reshape(1, -1)
    #print(f"Output samples shape: {samples.shape}")
    
    return samples


def resample_pcm16_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Resample flat mono signed-16-bit PCM without changing its representation."""
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise TypeError("samples must be a flat int16 array")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return samples
    target_size = max(1, round(samples.size * dst_rate / src_rate))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(target_size, dtype=np.float64) * src_rate / dst_rate
    resampled = np.interp(target_positions, source_positions, samples)
    return np.rint(resampled).clip(-32768, 32767).astype(np.int16)
