import argparse
import asyncio
from collections import deque
import threading

import numpy as np
import sounddevice as sd
from reactivex.subject import Subject

from mulive.core.stream_dsl import (
    Stream,
    SubGroup,
    drop_while,
    final_transcript_text,
    print_sink,
    turn_detector,
    stt,
)
from mulive.core.tts_providers import KokoroFastApiTTSProvider, PlaybackState, tts_sink


AEC_SAMPLE_RATES = (16000, 32000, 48000)
_AUDIO_STATUS_FIELDS = (
    "input_overflow",
    "input_underflow",
    "output_overflow",
    "output_underflow",
    "priming_output",
)


class AudioStatusTracker:
    """Collect PortAudio status flags without printing from the audio callback."""

    def __init__(self):
        self._counts = {field: 0 for field in _AUDIO_STATUS_FIELDS}
        self._lock = threading.Lock()

    def record(self, status) -> None:
        flags = [field for field in _AUDIO_STATUS_FIELDS if getattr(status, field, False)]
        if not flags:
            return
        with self._lock:
            for field in flags:
                self._counts[field] += 1

    def consume(self) -> dict[str, int]:
        with self._lock:
            counts = {field: count for field, count in self._counts.items() if count}
            self._counts = {field: 0 for field in _AUDIO_STATUS_FIELDS}
        return counts


class VadFrameBuffer:
    """Accumulate AEC frames into fixed-size windows for the turn detector."""

    def __init__(self, sample_rate: int, window_ms: int = 200):
        self.window_samples = sample_rate * window_ms // 1000
        self._buffer = np.empty(self.window_samples, dtype=np.int16)
        self._size = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Return each complete VAD window assembled from cleaned AEC frames."""
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        windows = []
        while samples.size:
            take = min(self.window_samples - self._size, samples.size)
            self._buffer[self._size : self._size + take] = samples[:take]
            self._size += take
            samples = samples[take:]
            if self._size == self.window_samples:
                windows.append(self._buffer.copy())
                self._size = 0
        return windows

    def clear(self) -> None:
        self._size = 0


class DuplexPcmSink:
    """Thread-safe TTS PCM queue shared by the TTS worker and PortAudio callback."""

    def __init__(self, loop: asyncio.AbstractEventLoop, sample_rate: int, max_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self._loop = loop
        self._max_samples = int(sample_rate * max_seconds)
        self._chunks = deque()
        self._queued_samples = 0
        self._source_tail = np.empty(0, dtype=np.int16)
        self._lock = threading.Lock()
        self._drained = asyncio.Event()
        self._space_available = asyncio.Event()
        self._drained.set()
        self._space_available.set()

    @staticmethod
    def _resample_24k_to_16k(samples: np.ndarray, tail: np.ndarray):
        """Resample exactly 24 kHz PCM to 16 kHz while retaining chunk continuity."""
        source = np.concatenate((tail, np.asarray(samples, dtype=np.int16).reshape(-1)))
        usable = source.size - (source.size % 3)
        triples = source[:usable].reshape(-1, 3)
        output = np.empty(triples.shape[0] * 2, dtype=np.int16)
        output[0::2] = triples[:, 0]
        output[1::2] = ((triples[:, 1].astype(np.int32) + triples[:, 2].astype(np.int32)) // 2).astype(np.int16)
        return output, source[usable:].copy()

    async def write_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            if (sample_rate, self.sample_rate) != (24000, 16000):
                raise ValueError(f"Unsupported TTS resampling: {sample_rate}Hz -> {self.sample_rate}Hz")
            with self._lock:
                samples, self._source_tail = self._resample_24k_to_16k(samples, self._source_tail)
        else:
            samples = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
        if not samples.size:
            return
        await self._enqueue(samples)

    async def _enqueue(self, samples: np.ndarray) -> None:
        """Queue PCM without dropping audio when synthesis outruns playback."""
        offset = 0
        while offset < samples.size:
            with self._lock:
                available = self._max_samples - self._queued_samples
                if available:
                    take = min(available, samples.size - offset)
                    self._chunks.append(samples[offset : offset + take].copy())
                    self._queued_samples += take
                    self._drained.clear()
                    offset += take
                    if self._queued_samples == self._max_samples:
                        self._space_available.clear()
                    continue
                self._space_available.clear()
            await self._space_available.wait()

    def render(self, frames: int) -> np.ndarray:
        """Return the precise speaker frame and consume it from the playback queue."""
        output = np.zeros(frames, dtype=np.int16)
        with self._lock:
            offset = 0
            while offset < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - offset, chunk.size)
                output[offset : offset + take] = chunk[:take]
                offset += take
                self._queued_samples -= take
                if take == chunk.size:
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[take:]
            if self._queued_samples < self._max_samples:
                self._loop.call_soon_threadsafe(self._mark_space_available)
            if self._queued_samples == 0:
                self._loop.call_soon_threadsafe(self._mark_drained_if_empty)
        return output

    def _mark_space_available(self) -> None:
        with self._lock:
            if self._queued_samples < self._max_samples:
                self._space_available.set()

    def _mark_drained_if_empty(self) -> None:
        with self._lock:
            if self._queued_samples == 0:
                self._drained.set()

    async def wait_until_played(self) -> None:
        while True:
            with self._lock:
                if self._queued_samples == 0:
                    return
            await self._drained.wait()

    async def finish(self) -> None:
        """Flush the sub-10 ms source tail at the end of a TTS utterance."""
        with self._lock:
            if not self._source_tail.size:
                return
            tail = self._source_tail[:1].copy()
            self._source_tail = np.empty(0, dtype=np.int16)
        await self._enqueue(tail)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._queued_samples = 0
            self._source_tail = np.empty(0, dtype=np.int16)
        self._loop.call_soon_threadsafe(self._drained.set)
        self._loop.call_soon_threadsafe(self._space_available.set)


async def run_mic_pipeline(args):
    loop = asyncio.get_running_loop()
    audio_input = Subject()
    subs = SubGroup()
    playback_state = PlaybackState() if args.tts else None
    mic_muted = threading.Event()
    stop_event = asyncio.Event()
    audio_status = AudioStatusTracker()
    pcm_sink = DuplexPcmSink(loop, args.sample_rate) if args.tts else None

    audio = Stream.source(audio_input, name="mic_audio")
    turn = audio | turn_detector()
    audio = turn.value
    if playback_state is not None and args.allow_interruptions and not args.aec:
        audio = audio | drop_while(
            lambda: playback_state.is_playing_or_recent(args.echo_suppress_seconds),
            name="drop_tts_feedback",
        )
    transcripts = audio | stt(
        provider=args.stt_provider,
        model=args.stt_model,
        model_size=args.model_size,
    )
    user_text = transcripts | final_transcript_text()

    user_text.to(print_sink(prefix="User: "), name="print_user_text", subs=subs)

    if args.tts:
        user_text.to(
            tts_sink(
                KokoroFastApiTTSProvider(pcm_sink=pcm_sink),
                interrupts=turn.started,
                name="kokoro_tts",
                state=playback_state,
            ),
            name="kokoro_tts",
            subs=subs,
        )
        if args.allow_interruptions and args.aec:
            print("TTS enabled with acoustic echo cancellation; VAD and STT remain active during playback.")
        elif args.allow_interruptions:
            print(f"TTS enabled. VAD stays active; STT segments are dropped during playback and for {args.echo_suppress_seconds}s after.")
        else:
            print("TTS enabled. Mic packets are muted during playback.")


    processor = None
    vad_buffer = None
    if args.aec:
        from pywebrtc_audio import AudioProcessor

        processor = AudioProcessor(
            sample_rate=args.sample_rate,
            echo_cancellation=True,
            noise_suppression=args.noise_suppression,
            auto_gain_control=args.auto_gain_control,
            stream_delay_ms=args.aec_delay_ms,
        )
        vad_buffer = VadFrameBuffer(args.sample_rate)

    def process_input(indata, frames, status):
        if status:
            audio_status.record(status)
        far = pcm_sink.render(frames) if pcm_sink is not None else np.zeros(frames, dtype=np.int16)
        if mic_muted.is_set():
            if vad_buffer is not None:
                vad_buffer.clear()
            return far
        if playback_state is not None and not args.allow_interruptions and playback_state.is_playing():
            if vad_buffer is not None:
                vad_buffer.clear()
            return far
        samples = indata[:, 0].copy()
        if processor is not None:
            samples = processor.process(samples, far)
        # AEC consumes 10 ms frames, while the existing Silero VAD needs a
        # larger window to classify speech reliably. The audio callback stays
        # real-time sized; only delivery to the VAD/STT pipeline is batched.
        vad_windows = vad_buffer.push(samples) if vad_buffer is not None else [samples]
        for window in vad_windows:
            loop.call_soon_threadsafe(audio_input.on_next, window)
        return far

    # Full-duplex callback: its rendered speaker frame is the AEC far-end reference.
    def duplex_callback(indata, outdata, frames, time_info, status):
        outdata[:, 0] = process_input(indata, frames, status)

    def input_callback(indata, frames, time_info, status):
        process_input(indata, frames, status)

    async def keyboard_controls():
        while not stop_event.is_set():
            command = (await asyncio.to_thread(input, "")).strip().lower()
            if command == "m":
                if mic_muted.is_set():
                    mic_muted.clear()
                    print("Mic unmuted.")
                else:
                    mic_muted.set()
                    print("Mic muted.")
            elif command == "q":
                stop_event.set()

    controls_task = asyncio.create_task(keyboard_controls())

    print("Listening. Speak, then pause for transcription.")
    print("Controls: m + Enter toggles mic mute, q + Enter stops, Ctrl-C also stops.")
    try:
        stream_type = sd.Stream if args.tts else sd.InputStream
        callback = duplex_callback if args.tts else input_callback
        stream_args = {
            "samplerate": args.sample_rate,
            "channels": 1,
            "dtype": "int16",
            "blocksize": args.block_size,
            "callback": callback,
        }
        if args.tts and (args.input_device is not None or args.output_device is not None):
            stream_args["device"] = (args.input_device, args.output_device)
        elif args.input_device is not None:
            stream_args["device"] = args.input_device
        with stream_type(**stream_args):
            while not stop_event.is_set():
                await asyncio.sleep(0.25)
                status_counts = audio_status.consume()
                if status_counts:
                    details = ", ".join(f"{name}={count}" for name, count in status_counts.items())
                    print(f"audio status: {details}")
    finally:
        stop_event.set()
        controls_task.cancel()
        audio_input.on_completed()
        subs.dispose()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the basic DSL pipeline from local microphone input.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, help="Audio callback size; defaults to a 10 ms AEC frame.")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--stt-provider", default="mlx")
    parser.add_argument("--stt-model")
    parser.add_argument("--tts", action="store_true", help="Speak transcribed text with Kokoro TTS.")
    parser.add_argument("--echo-suppress-seconds", type=float, default=2.0)
    parser.add_argument("--allow-interruptions", action="store_true", help="Keep VAD active during TTS so speech can interrupt playback.")
    parser.add_argument("--no-aec", dest="aec", action="store_false", help="Disable WebRTC echo cancellation and use legacy TTS suppression.")
    parser.add_argument("--aec-delay-ms", type=int, default=40, help="Initial speaker-to-microphone delay hint for AEC.")
    parser.add_argument("--noise-suppression", action="store_true", help="Enable WebRTC noise suppression with AEC.")
    parser.add_argument("--auto-gain-control", action="store_true", help="Enable WebRTC automatic gain control with AEC.")
    parser.add_argument("--input-device", type=int, help="PortAudio input device index; defaults to the system input device.")
    parser.add_argument("--output-device", type=int, help="PortAudio output device index for TTS; defaults to the system output device.")
    parser.set_defaults(aec=True)
    args = parser.parse_args()
    if args.aec and args.sample_rate not in AEC_SAMPLE_RATES:
        parser.error(f"--sample-rate must be one of {AEC_SAMPLE_RATES} when AEC is enabled")
    if args.tts and args.sample_rate != 16000:
        parser.error("--tts currently requires --sample-rate 16000 (Kokoro PCM is bridged from 24 kHz)")
    if args.output_device is not None and not args.tts:
        parser.error("--output-device requires --tts")
    frame_samples = args.sample_rate // 100
    if args.block_size is None:
        args.block_size = frame_samples
    if args.aec and args.block_size % frame_samples:
        parser.error(f"--block-size must be a multiple of {frame_samples} for 10 ms AEC frames")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(run_mic_pipeline(parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")
