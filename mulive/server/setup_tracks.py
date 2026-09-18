from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
import asyncio
import json
import os

from ..core.audio_utils import convert_and_resample_frame
from ..core.audio_input import SubjectAudioInput
from ..core.audio_output import AudioChunk
from ..core.webrtc_audio import AssistantAudioTrack
from ..core.session import SessionContext
from ..core.utils import rx_Subject as Subject # for input audio/video subjects
import numpy as np


DEFAULT_STUN_URLS = "stun:stun.l.google.com:19302"


def _load_ice_servers(config):
    """Build the ICE-server list for the peer connection.

    On EC2 (and any 1:1 NAT box) aiortc only sees the private IP, so the
    remote browser cannot reach it. A STUN server lets aiortc discover its
    public reflexive address and hand that to the client. On desktop STUN
    is inert, so this is a strict superset of the previous behaviour.

    Sources, in priority order:

    1. ``config['ice_servers']`` — a list of dicts ``{urls, username?, credential?}``.
    2. ``STUN_URLS`` env var — comma-separated STUN/TURN URLs.
    3. Google's public STUN as a safe default.
    """
    configured = (config or {}).get("ice_servers")
    if configured:
        return [RTCIceServer(**server) for server in configured]
    urls = os.environ.get("STUN_URLS", DEFAULT_STUN_URLS).strip()
    if not urls:
        return []
    return [RTCIceServer(urls=[u.strip() for u in urls.split(",") if u.strip()])]

async def setup_audio_track(pc, track: MediaStreamTrack, speech_turn_input, stop_event, config):
    # ISC: R1 R2 T1 T2 I_SAFE I_AUTH I_LIVE I_FRESH I_ATOMIC
    """Handle incoming audio track and process it through the pipeline.

    Every buffered chunk is forwarded straight to the speech-turn subject.
    Silero VAD downstream is the real speech gate — it already rejects
    silence cheaply (~1 ms per 500 ms chunk on onnxruntime), so a second
    RMS pre-filter here just spent CPU on a 1-vCPU box and printed
    per-chunk debug lines that noised up the logs.
    """
    audio_buffer_size = config.get("input_audio_buffer_size", 8000)

    buffer = np.array([], dtype=np.int16)
    bsize = audio_buffer_size #audio buffer size, for sending to pipeline
    sr = 16000

    while not stop_event.is_set():
        try:
            frame = await track.recv()
            chunk = convert_and_resample_frame(frame, target_sample_rate=sr)
            buffer = np.concatenate([buffer, chunk])

            # Process complete chunks
            while len(buffer) >= bsize:
                speech_turn_input.write(AudioChunk(buffer[:bsize], sample_rate=sr))
                buffer = buffer[bsize:]
                
        except MediaStreamError:
            print("Audio track ended")
            stop_event.set()
            break
        except Exception as e:
            import traceback
            print(f"Error processing audio: {e}")
            print(f"StackTrace: {traceback.format_exc()}")

            stop_event.set()
            break
    
    # Process any remaining audio
    if len(buffer) > 0 and not stop_event.is_set():
        speech_turn_input.write(AudioChunk(buffer, sample_rate=sr))
    speech_turn_input.close()
    
    print("Audio processing stopped")

async def setup_video_track(pc, track: MediaStreamTrack, video_obs_input, stop_event, config):
    
    """Handle incoming video track. Sample at interval, send to observer"""
    input_video_sample_interval = config.get("input_video_sample_interval", 500)
    
    frame_count = 0
    
    while not stop_event.is_set():
        try:
            frame = await track.recv()
            frame_count += 1
            
            # Log frame info occasionally
            if frame_count % input_video_sample_interval == 0:
                if config.get("debug_video_frames", False):
                    print("sending frame")
                video_obs_input.on_next(frame)
                
        except MediaStreamError:
            print("Video track ended")
            video_obs_input.on_next(None)
            break
        except Exception as e:
            print(f"Error processing video: {e}")
            video_obs_input.on_next(None)
            break
    
    print("Video processing stopped")

def pc_session_setup(run_session, config, on_peer_close=None):

    pc = RTCPeerConnection(RTCConfiguration(iceServers=_load_ice_servers(config)))
    
    stop_event = asyncio.Event()
    data_channels = {}
    assistant_audio_track = AssistantAudioTrack()
    pc.assistant_audio_track = assistant_audio_track
    pc.addTrack(assistant_audio_track)
    main_loop = asyncio.get_running_loop()
    audio_input, video_input, client_input = SubjectAudioInput(Subject()), Subject(), Subject()
    session = SessionContext(
        pc=pc,
        audio_output=assistant_audio_track,
        data_channels=data_channels,
        audio_input=audio_input,
        video_input=video_input,
        client_input=client_input,
        main_loop=main_loop,
    )

    def update_ready():
        server_text = data_channels.get("server_text")
        if pc.connectionState == "connected" and server_text is not None and server_text.readyState == "open":
            session.ready.set()

    def on_datachannel(channel):
        print(f"Data channel received: {channel.label}")
        data_channels[channel.label] = channel

        @channel.on("open")
        def on_open():
            update_ready()

        @channel.on("close")
        def on_close():
            if channel.label == "server_text":
                stop_event.set()
                session.closed.set()
                if on_peer_close is not None:
                    on_peer_close(pc)
                if pc.connectionState != "closed":
                    main_loop.create_task(pc.close())

        @channel.on("message")
        def on_message(message):
            print("Message from client:", message)
            try:
                payload = json.loads(message) if isinstance(message, str) else message
                client_input.on_next(payload)
            except (TypeError, json.JSONDecodeError) as exc:
                print(f"Ignoring invalid client message: {exc}")

        update_ready()
    
    pc.on("datachannel", on_datachannel)

    # The session runner owns app behavior for this peer and can wait for
    # session.ready before producing its first output.
    session_task = asyncio.create_task(run_session(session))
    pc.session_context = session
    pc.session_task = session_task

    def on_session_done(task):
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        print(f"Session task failed: {type(exception).__name__}: {exception}")
        stop_event.set()
        session.closed.set()
        if on_peer_close is not None:
            on_peer_close(pc)
        if pc.connectionState != "closed":
            main_loop.create_task(pc.close())

    session_task.add_done_callback(on_session_done)
    
    def on_track(track: MediaStreamTrack):
        print(f"Track received: {track.kind}")
        # audio/video track --- (initial filters)--> audio/video input
        if track.kind == "audio":
            asyncio.create_task(
                setup_audio_track(pc, track, audio_input, stop_event, config)
            )
        elif track.kind == "video":
            asyncio.create_task(setup_video_track(pc, track, video_input, stop_event, config))
    
    pc.on("track", on_track)
    
    async def on_connectionstatechange():
        print(f"Connection state changed to: {pc.connectionState}")
        update_ready()
        if pc.connectionState in ["failed", "closed"]:
            stop_event.set()
            session.closed.set()
            if not session_task.done():
                session_task.cancel()
            if on_peer_close is not None:
                on_peer_close(pc)
            if pc.connectionState != "closed":
                await pc.close()
    
    pc.on("connectionstatechange", on_connectionstatechange)
    return pc
