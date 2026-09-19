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
- **Keep product logic explicit.** Voice, video, UI actions, and model responses
  can be handled with deterministic state transitions.
- **Run close to the product.** Use direct WebRTC and local-capable STT/TTS;
  choose cloud models only where they add value.
- **Make interactions testable.** Test a single utterance or a complete
  interaction, including interruption, cancellation, and application outcomes.

## Choose your path

- **Browser quickstart:** Run a complete WebRTC voice interaction in the bundled
  browser client. Start with [Quickstart](#quickstart).
- **Embedded FastAPI:** Add Mulive's browser client and voice transport to an
  existing application. See [Embed in FastAPI](#embed-in-fastapi).
- **Local microphone:** Run speech recognition and speech output without a
  browser. See [Local microphone](#local-microphone).

## Quickstart

A Mulive voice interaction has five parts:

```text
microphone → utterance detection → speech recognition → response → speech output
```

The response can be a direct stream transformation, normal application code,
or a model-backed `Agent`. Start with the direct path so each boundary is
visible.

This guide uses five terms consistently. An **utterance** is one piece of user
speech. A **transcript** is its recognized text. A **response** is the text the
application returns. A **client event** is structured data sent to the browser.
**Speech output** is the generated audio the user hears.

### Echo recognized speech

```bash
pip install mulive

# Browser audio + local voice stack
python -m mulive.quickstart.web --http --tts-browser
```

Open `http://localhost:9000` and start streaming. `localhost` is trusted by
modern browsers, so microphone access works without a certificate.
The browser sends real-time media to Python. It receives transcripts and speech
output. Without `GROQ_API_KEY`, the quickstart echoes each recognized utterance.
The core flow is:

```python
audio = Stream.source(session.audio_input, name="audio")
turn = audio | turn_detector()
transcripts = turn.value | stt(provider="faster_whisper", model_size="small")
user_text = transcripts | final_transcript_text()

user_text | to_user(session=session, role="user", subs=subs)
user_text | to_user(
    session=session, role="assistant", subs=subs,
    tts=tts_config, tts_provider=tts_provider, interrupts=turn.started,
)
```

`turn_detector()` groups microphone audio into utterances. `stt()` converts each
utterance into a transcript. The two `to_user()` connections show the transcript,
then return the same text as the response through the display and speech output.

### Let an LLM control responses

Use `Agent` when responses need conversation history and a language model. The
audio, utterance detection, speech recognition, and speech output stay the same.
Only the response step changes:

```python
from mulive.apps.agent import Agent

final_transcripts = transcripts | filter_items(lambda event: event.is_final)
user_text = final_transcripts | map_items(lambda event: event.text) | non_empty_text()

agent = Agent(llm_timeout_s=30)
agent.connect(final_transcripts, turn.started)

user_text | to_user(session=session, role="user", subs=subs)
agent.assistant_text | to_user(
    session=session, role="assistant", subs=subs,
    tts=tts_config, tts_provider=tts_provider, interrupts=turn.started,
)
agent.client_events.to(client_message_sink(session), subs=subs)
agent.start()
```

Install the Groq integration and set its key before running the same browser
quickstart:

```bash
pip install "mulive[groq]"
export GROQ_API_KEY="your-key"
python -m mulive.quickstart.web --http --tts-browser
```

`Agent` owns conversation history and runs one model call at a time. A new
utterance stops current speech output but does not cancel model work. If the
user asks another question while a call is running, the agent acknowledges it
and waits for the call or its 30-second timeout before answering the newer
utterance.

The web quickstart uses Faster-Whisper `small` for recognition. Add
`--tts-browser` or `--tts-local` to send speech output through the browser or
the server's speakers. Override the recognition model with `--model-size`, use
`--stt-provider mlx` on Apple Silicon, and change Groq's model with
`--llm-model`. Change the model timeout with `--llm-timeout-s`.

## Embed in FastAPI

Mount voice into an application you already own:

```bash
pip install "mulive[fastapi]"
```

```python
from fastapi import FastAPI
from mulive import mount_voice

app = FastAPI()
mount_voice(app)
```

Mulive serves its browser client and voice WebSocket transport from the same
application.

## Voice model installation options

Mulive includes portable local defaults: Faster-Whisper for speech recognition
and Piper for speech output. They run on CPU and download their selected model
weights on first use. `pip install mulive` is enough for the browser quickstart
without Groq.

| Option | Install | Use |
| --- | --- | --- |
| MLX Whisper (Apple Silicon) | `pip install "mulive[mlx]"` | `--stt-provider mlx` |
| Groq responses | `pip install "mulive[groq]"` | Set `GROQ_API_KEY` for the web quickstart |
| In-process Kokoro ONNX | `pip install "mulive[kokoro]"` | Select `kokoro_onnx` in an app's TTS configuration; not a web quickstart flag |
| External Kokoro-FastAPI | `pip install "mulive[openai]"` | Run [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) separately at `localhost:8880` (or set `KOKORO_FASTAPI_URL`) |
| Gemini models | `pip install "mulive[gemini]"` | Select Gemini in an app that supports it; the web quickstart uses Groq only |

Extras can be combined, for example `pip install "mulive[mlx,groq]"`.
`TTSConfig()` defaults to `kokoro_onnx`, so install `mulive[kokoro]` when
using that default. The browser quickstart explicitly selects Piper.

### Local microphone

The local microphone example uses WebRTC acoustic echo cancellation (AEC) by
default. With `--tts --allow-interruptions`, it uses the speech output as the
echo reference. You can speak while the assistant is speaking. Start
Kokoro-FastAPI first, then:

```bash
pip install "mulive[local-audio,openai]"
python -m mulive.quickstart.mic --stt-provider faster_whisper --model-size small --tts --allow-interruptions
```

For MLX on Apple Silicon, add the `mlx` extra and use `--stt-provider mlx`.
Use `--no-aec` only if you need to disable echo cancellation.

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
| [Microphone loop](mulive/quickstart/mic.py) | Local microphone, echo cancellation, utterance detection, speech recognition, and optional speech output | Available |
| [Browser audio](mulive/quickstart/web.py) | WebRTC audio, transcript echo or Groq response, and browser speech output | Available |
| [Math helper](mulive/quickstart/multimodal_agent.py) | Browser audio/video, latest-frame vision, and spoken responses | Experimental |
| Audio todo app | Voice-driven UI state and actions | Planned |
| Market voice dashboard | Voice control plus live visual data | Planned |
| Card-cancellation flow | Guarded voice workflow with explicit confirmation | Planned |

## The application model

Mulive separates the parts that benefit from models from the parts your product
must control:

```text
mic / camera / browser UI
            ↓
  transcripts + client events
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
audio → utterance detection → speech recognition → model → speech output
```

The same application model is intended to support direct streaming
speech-to-speech models as they are added. The product state and UI should not
need to change just because the voice model does.

## What is next

| Capability | Direction |
| --- | --- |
| Embeddable web voice | A small browser client and stable server integration for existing web apps |
| Backend speech output | Stream generated speech from a Python service without the browser UI |
| Systematic evaluation | Utterance-level assertions and end-to-end interaction scenarios |
| Latency explorer | Per-utterance timing from speech end through playback |
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
