const PROTOCOL = "mulive.voice.v1";
const INPUT_RATE = 16_000;
const PLAYBACK_BUFFER_SECONDS = 0.12;

function clientAssetURL(name) {
  return new URL(name, import.meta.url).href;
}

function websocketURL(path) {
  if (/^wss?:\/\//.test(path)) return path;
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}${path}`;
}

function pcm16(samples, sourceRate) {
  const count = Math.max(1, Math.round(samples.length * INPUT_RATE / sourceRate));
  const output = new Int16Array(count);
  for (let index = 0; index < count; index += 1) {
    const start = Math.floor(index * sourceRate / INPUT_RATE);
    const end = Math.min(samples.length, Math.floor((index + 1) * sourceRate / INPUT_RATE));
    let total = 0;
    for (let sample = start; sample < Math.max(start + 1, end); sample += 1) total += samples[sample];
    output[index] = Math.max(-1, Math.min(1, total / Math.max(1, end - start))) * 32767;
  }
  return output.buffer;
}

export class VoiceClient {
  constructor(path, { mode = "ptt", onPcm = null } = {}) {
    if (!["ptt", "vad"].includes(mode)) throw new Error("mode must be 'ptt' or 'vad'");
    this.path = path;
    this.mode = mode;
    this.onPcm = onPcm;
    this.ws = null;
    this.listeners = new Map();
    this.audioContext = null;
    this.mediaStream = null;
    this.captureNode = null;
    this.playbackNode = null;
    this.captureModulePromise = null;
    this.playbackModulePromise = null;
    this.turnID = null;
    this.captureTurn = mode === "vad";
    this.outputRate = 24_000;
  }

  static async connect(path, options = {}) {
    const { playback = false, ...clientOptions } = options;
    const client = new VoiceClient(path, clientOptions);
    const playbackReady = playback ? client.ensurePlayback() : Promise.resolve();
    try {
      await Promise.all([client.connect(), playbackReady]);
      return client;
    } catch (error) {
      await client.disconnect();
      throw error;
    }
  }

  on(type, handler) {
    const handlers = this.listeners.get(type) || new Set();
    handlers.add(handler);
    this.listeners.set(type, handlers);
    return () => handlers.delete(handler);
  }

  emit(type, event) {
    for (const handler of this.listeners.get(type) || []) handler(event);
  }

  async connect() {
    if (this.ws) return;
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(websocketURL(this.path));
      ws.binaryType = "arraybuffer";
      ws.onerror = () => reject(new Error("Mulive WebSocket connection failed"));
      ws.onopen = () => ws.send(JSON.stringify({ type: "session.hello", protocol: PROTOCOL, mode: this.mode }));
      ws.onmessage = (message) => this.handleMessage(message, resolve);
      ws.onclose = () => this.emit("closed", {});
      this.ws = ws;
    });
  }

  handleMessage(message, connected) {
    if (typeof message.data !== "string") {
      this.playPCM(message.data);
      return;
    }
    const event = JSON.parse(message.data);
    if (event.type === "session.ready") connected?.(event);
    if (event.type === "response.started") this.clearPlayback();
    if (event.type === "response.audio" && event.sample_rate) {
      this.outputRate = event.sample_rate;
      this.playbackNode?.port.postMessage({
        type: "configure",
        sampleRate: this.outputRate,
        prebufferSeconds: PLAYBACK_BUFFER_SECONDS,
      });
    }
    if (event.type === "response.cancelled") this.clearPlayback();
    this.emit(event.type, event);
  }

  async start() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error("voice is not connected");
    if (this.mediaStream) return;
    const audioContext = await this.ensureAudioContext();
    await this.ensurePlayback();
    if (!this.captureModulePromise) {
      this.captureModulePromise = audioContext.audioWorklet.addModule(clientAssetURL("pcm-worklet.js"));
    }
    await this.captureModulePromise;
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(this.mediaStream);
    this.captureNode = new AudioWorkletNode(audioContext, "mulive-pcm");
    this.captureNode.port.onmessage = ({ data }) => {
      if (!this.captureTurn || this.ws?.readyState !== WebSocket.OPEN) return;
      const payload = pcm16(data, audioContext.sampleRate);
      this.onPcm?.(payload);
      this.ws.send(payload);
    };
    source.connect(this.captureNode);
    this.captureNode.connect(audioContext.destination);
  }

  async ensureAudioContext() {
    if (!this.audioContext) this.audioContext = new AudioContext();
    if (this.audioContext.state === "suspended") await this.audioContext.resume();
    return this.audioContext;
  }

  async ensurePlayback() {
    const audioContext = await this.ensureAudioContext();
    if (this.playbackNode) return;
    if (!this.playbackModulePromise) {
      this.playbackModulePromise = audioContext.audioWorklet.addModule(clientAssetURL("pcm-player-worklet.js"));
    }
    await this.playbackModulePromise;
    if (this.playbackNode) return;
    this.playbackNode = new AudioWorkletNode(audioContext, "mulive-pcm-player", {
      outputChannelCount: [1],
    });
    this.playbackNode.connect(audioContext.destination);
    this.playbackNode.port.postMessage({
      type: "configure",
      sampleRate: this.outputRate,
      prebufferSeconds: PLAYBACK_BUFFER_SECONDS,
    });
  }

  beginTurn() {
    if (this.mode !== "ptt") throw new Error("beginTurn is only available in PTT mode");
    if (!this.mediaStream) throw new Error("call start() before beginTurn()");
    this.turnID = crypto.randomUUID();
    this.captureTurn = true;
    this.send({ type: "turn.start", turn_id: this.turnID });
  }

  endTurn() {
    if (!this.turnID) return;
    const turnID = this.turnID;
    this.turnID = null;
    this.captureTurn = false;
    this.send({ type: "turn.commit", turn_id: turnID });
  }

  cancelTurn() {
    if (!this.turnID) return;
    const turnID = this.turnID;
    this.turnID = null;
    this.captureTurn = false;
    this.send({ type: "turn.cancel", turn_id: turnID });
  }

  async speak(text) {
    await this.ensurePlayback();
    this.send({ type: "tts.speak", text });
  }
  cancelSpeech() { this.send({ type: "tts.cancel" }); }

  send(event) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error("voice is not connected");
    this.ws.send(JSON.stringify(event));
  }

  sendPCM(buffer) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error("voice is not connected");
    this.ws.send(buffer);
  }

  playPCM(buffer) {
    if (!this.playbackNode) return;
    this.playbackNode.port.postMessage({ type: "pcm", buffer }, [buffer]);
  }

  clearPlayback() {
    this.playbackNode?.port.postMessage({ type: "clear" });
  }

  async stop() {
    this.cancelTurn();
    this.captureNode?.disconnect();
    this.captureNode = null;
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
  }

  async disconnect() {
    await this.stop();
    this.clearPlayback();
    this.playbackNode?.disconnect();
    this.playbackNode = null;
    await this.audioContext?.close();
    this.audioContext = null;
    this.captureModulePromise = null;
    this.playbackModulePromise = null;
    this.ws?.close();
    this.ws = null;
  }
}

export const Mulive = { connect: VoiceClient.connect };
