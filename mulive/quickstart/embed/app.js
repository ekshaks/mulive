import { Mulive } from "/mulive/client/mulive.js";

// Find the controls rendered by index.html.
const status = document.querySelector("#status");
const statusText = document.querySelector("#statusText");
const events = document.querySelector("#events");
const mode = document.querySelector("#mode");
const connect = document.querySelector("#connect");
const talk = document.querySelector("#talk");
const speakForm = document.querySelector("#speak-form");
const speechText = document.querySelector("#speech-text");
const speak = document.querySelector("#speak");

// Share one voice session between listening and speaking.
let voice;
let voicePromise;
let microphoneConnected = false;

// Show useful protocol events, newest first.
function log(event) {
  if (events.textContent === "Events will appear here.") events.textContent = "";
  events.textContent = `${JSON.stringify(event)}\n${events.textContent}`;
}

// Connect once on demand and attach the demo event listeners.
async function getVoice() {
  if (voice) return voice;
  if (!voicePromise) {
    voicePromise = Mulive.connect("/mulive-ws", { mode: mode.value, playback: true })
      .then((client) => {
        voice = client;
        mode.disabled = true;
        client.on("transcript.final", log);
        client.on("response.text", log);
        client.on("error", log);
        client.on("closed", () => {
          if (voice && voice !== client) return;
          voice = undefined;
          voicePromise = undefined;
          microphoneConnected = false;
          mode.disabled = false;
          connect.disabled = false;
          connect.textContent = "Connect microphone";
          talk.disabled = true;
          status.dataset.state = "closed";
          statusText.textContent = "Disconnected";
        });
        return client;
      })
      .catch((error) => {
        voicePromise = undefined;
        throw error;
      });
  }
  return voicePromise;
}

// Toggle microphone capture; disconnecting also unlocks mode selection.
connect.addEventListener("click", async () => {
  if (microphoneConnected) {
    connect.disabled = true;
    try {
      const client = voice;
      await client.disconnect();
      if (voice === client) {
        voice = undefined;
        voicePromise = undefined;
      }
      microphoneConnected = false;
      mode.disabled = false;
      connect.textContent = "Connect microphone";
      talk.disabled = true;
      status.dataset.state = "idle";
      statusText.textContent = "Microphone disconnected. Choose a listening mode or use text to speech.";
    } finally {
      connect.disabled = false;
    }
    return;
  }

  try {
    status.dataset.state = "connecting";
    statusText.textContent = "Connecting…";
    connect.disabled = true;
    voice = await getVoice();
    await voice.start();
    microphoneConnected = true;
    status.dataset.state = "connected";
    statusText.textContent = mode.value === "ptt" ? "Hold the button to speak." : "Listening continuously.";
    connect.disabled = false;
    connect.textContent = "Disconnect microphone";
    talk.disabled = mode.value !== "ptt";
  } catch (error) {
    status.dataset.state = "error";
    statusText.textContent = `Could not start voice: ${error.message}`;
    microphoneConnected = false;
    connect.textContent = "Connect microphone";
    connect.disabled = false;
  }
});

// Push-to-talk maps pointer gestures to one explicit speech turn.
talk.addEventListener("pointerdown", () => voice.beginTurn());
talk.addEventListener("pointerup", () => voice.endTurn());
talk.addEventListener("pointercancel", () => voice.cancelTurn());
talk.addEventListener("pointerleave", () => voice.cancelTurn());

// TTS connects independently and never requests microphone permission.
speakForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = speechText.value.trim();
  if (!text) return;
  const originalLabel = speak.textContent;
  speak.disabled = true;
  speak.textContent = "Connecting…";
  try {
    const client = await getVoice();
    speak.textContent = "Speaking…";
    await client.speak(text);
  } catch (error) {
    log({ type: "error", text: `Could not speak: ${error.message}` });
  } finally {
    speak.disabled = false;
    speak.textContent = originalLabel;
  }
});
