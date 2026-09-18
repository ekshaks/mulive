# Mulive Embed

Add STT and TTS to an existing FastAPI app.

## Run the demo

From the repository root:

```bash
PYTHONPATH=. uvicorn quickstart.embed.app:app --reload
```

Open `http://127.0.0.1:8000`.

## Use in your FastAPI app

Mount Mulive on your existing app:

```python
from fastapi import FastAPI
from mulive import mount_voice

app = FastAPI()
mount_voice(app)
```

Defaults:

- WebSocket: `/mulive-ws`
- Browser SDK: `/mulive/client/mulive.js`

Override them when needed:

```python
mount_voice(
    app,
    websocket_path="/speech-ws",
    client_path="/speech/client",
)
```

Then import the browser client from any module on your page:

```js
import { Mulive } from "/mulive/client/mulive.js";

const voice = await Mulive.connect("/mulive-ws");
voice.on("transcript.final", ({ text }) => console.log(text));

listenButton.onclick = () => voice.start();
speakButton.onclick = () => voice.speak("Hello!");
```

`voice.start()` is the only operation that requests microphone access. TTS
works independently through `voice.speak(text)`.

## Configuration

Edit `config.yml` using the same `stt:` and `tts:` sections used by Mulive app
bundles. The demo uses faster-whisper tiny, ONNX Silero VAD, and Piper. Use
HTTPS outside localhost so browsers allow microphone access.
