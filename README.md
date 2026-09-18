<p align="center">
  <img src="https://raw.githubusercontent.com/ekshaks/mulive/main/docs/assets/mulive-hero.png" alt="Mulive: a Python voice application connected to a live browser interface and latency trace" width="1280" />
</p>

# Mulive

**Build local-first, interactive voice and multimodal applications in Python.**

Mulive is a lightweight real-time substrate that connects microphone, camera,
models, application state, and web UI. Build a tiny voice interaction first,
then grow it into a full product without replacing the runtime underneath.

> Voice is an interface to your application—not the application itself.

## Why Mulive

- **Build applications, not just chatbots.** Voice can operate a game, tutor,
  workflow, dashboard, or custom web UI.
- **Keep product logic explicit.** Voice, video, UI actions, and model results
  can be handled as typed events with deterministic state transitions.
- **Run close to the product.** Use direct WebRTC and local-capable STT/TTS;
  choose cloud models only where they add value.
- **Make interactions testable.** Test a single turn or a complete interaction,
  including interruption, cancellation, and application outcomes.

## Start with a voice interaction

```bash
pip install mulive

# Browser audio + local voice stack
python -m mulive.quickstart.web --http --tts-browser
```

Open `http://localhost:9000` and start streaming. `localhost` is trusted by
modern browsers, so microphone access works without a certificate.
The browser sends real-time media to Python and receives text and audio
responses.

An audio-only browser pipeline is just a few composed stages:

```python
audio = Stream.source(session.audio_input, name="audio")
turn = audio | turn_detector()
transcripts = turn.segments | stt(provider="faster_whisper", model_size="tiny")
user_text = transcripts | final_transcript_text()

add_text_sinks(user_text, session, role="user", subs=subs)
add_tts(user_text, session.audio_output, turn.signals, subs=subs)
```

Run the complete version with:

```bash
python -m mulive.quickstart.web --http --tts-browser
```

## Choose a server setup

Use one of these public paths:

1. **Standalone browser app.** Run the WebRTC quickstart above for a complete
   browser client, signaling server, and local voice pipeline.
2. **Existing FastAPI app.** Mount voice into an app you already own:

   ```bash
   pip install "mulive[fastapi]"
   ```

   ```python
   from fastapi import FastAPI
   from mulive import mount_voice

   app = FastAPI()
   mount_voice(app)
   ```

   Mulive serves its small browser client and its voice WebSocket transport.
   There is no separate WebSocket server setup to choose.

## Default voice stack

Mulive includes portable local defaults: Faster-Whisper for speech recognition
and Piper for speech output. They run on CPU and download their selected model
weights on first use.

Use provider extras only for alternatives such as MLX, cloud models, or Kokoro.

### HTTPS for another device or deployment

Use HTTP only when the browser and Mulive run on the same machine. For a phone,
LAN host, public hostname, or an HTTPS app embedding Mulive, serve it over
HTTPS. For local device testing, create a trusted development certificate with
[`mkcert`](https://github.com/FiloSottile/mkcert):

```bash
mkcert -install
mkdir -p ~/.config/mulive/certs
mkcert -key-file ~/.config/mulive/certs/key.pem \
  -cert-file ~/.config/mulive/certs/cert.pem \
  localhost 127.0.0.1 ::1 <your-lan-hostname-or-ip>

python -m mulive.quickstart.web --tts-browser
```

For production, terminate TLS with the deployment platform or reverse proxy and
set `MULIVE_SSL_KEYFILE` and `MULIVE_SSL_CERTFILE` when Mulive should serve TLS
itself.

## What you can build

| Example | What it demonstrates | Status |
| --- | --- | --- |
| [Microphone loop](quickstart/mic.py) | Local microphone, turn detection, STT, and optional TTS | Available |
| [Browser audio](quickstart/web.py) | WebRTC audio, transcripts, and browser audio output | Available |
| [Math helper](quickstart/multimodal_agent.py) | Browser audio/video, latest-frame vision, and spoken responses | Experimental |
| Audio todo app | Voice-driven UI state and actions | Planned |
| Market voice dashboard | Voice control plus live visual data | Planned |
| Card-cancellation flow | Guarded voice workflow with explicit confirmation | Planned |

## The application model

Mulive separates the parts that benefit from models from the parts your product
must control:

```text
mic / camera / browser UI
            ↓
   streams and typed events
            ↓
  model calls + explicit app state
            ↓
 speech, UI updates, and safe effects
```

Use models for perception and conversation. Keep scoring, workflow state,
permissions, retries, and external effects in normal application code. That is
especially useful for tutoring, games, customer workflows, and any interaction
where an answer must be checked before the app moves on.

## Model architectures

Today, Mulive's examples use a cascaded pipeline:

```text
audio → turn detection → STT → LLM/VLM → TTS
```

The same application model is intended to support direct streaming
speech-to-speech models as they are added. The product state and UI should not
need to change just because the voice model does.

## What is next

| Capability | Direction |
| --- | --- |
| Embeddable web voice | A small browser client and stable server integration for existing web apps |
| Backend TTS | Stream generated speech from a Python service without the browser UI |
| Systematic evaluation | Turn-level assertions and end-to-end interaction scenarios |
| Latency explorer | Per-turn timing from speech end through playback |
| Streaming voice models | Direct speech-to-speech and hybrid model adapters |

## Project structure

```text
mulive/      Runtime, browser assets, and runnable quickstarts
mulive/core/ Real-time transport, streams, and media helpers
mulive/apps/ Application helpers
mulive/client/ Browser WebRTC client and UI
mulive/quickstart/ Small runnable applications
tests/       Pipeline, controller, transport, and application tests
docs/        Architecture and design notes
```

## Current status

Mulive is actively evolving. The available quickstarts are useful foundations;
the embed API, latency explorer, streaming-model adapters, and polished product
demos are intentionally marked as planned rather than presented as shipped.
