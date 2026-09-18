const DEFAULT_PREBUFFER_SECONDS = 0.12;

class MulivePCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceRate = 24_000;
    this.prebufferSeconds = DEFAULT_PREBUFFER_SECONDS;
    this.queue = [];
    this.queueOffset = 0;
    this.bufferedSamples = 0;
    this.playing = false;
    this.currentSample = 0;
    this.nextSample = 0;
    this.phase = 0;
    this.hasPair = false;

    this.port.onmessage = ({ data }) => {
      if (data?.type === "configure") {
        if (Number.isFinite(data.sampleRate) && data.sampleRate > 0) {
          this.sourceRate = data.sampleRate;
        }
        if (Number.isFinite(data.prebufferSeconds) && data.prebufferSeconds >= 0) {
          this.prebufferSeconds = data.prebufferSeconds;
        }
        return;
      }
      if (data?.type === "clear") {
        this.clear();
        return;
      }
      if (data?.type === "pcm" && data.buffer instanceof ArrayBuffer) {
        const samples = new Int16Array(data.buffer);
        if (samples.length) {
          this.queue.push(samples);
          this.bufferedSamples += samples.length;
        }
      }
    };
  }

  clear() {
    this.queue.length = 0;
    this.queueOffset = 0;
    this.bufferedSamples = 0;
    this.playing = false;
    this.phase = 0;
    this.hasPair = false;
  }

  readSample() {
    while (this.queue.length) {
      const chunk = this.queue[0];
      if (this.queueOffset < chunk.length) {
        const value = chunk[this.queueOffset] / 32_768;
        this.queueOffset += 1;
        this.bufferedSamples -= 1;
        return value;
      }
      this.queue.shift();
      this.queueOffset = 0;
    }
    return null;
  }

  startIfBuffered() {
    if (this.playing) return true;
    if (this.bufferedSamples / this.sourceRate < this.prebufferSeconds) return false;
    const current = this.readSample();
    const next = this.readSample();
    if (current === null || next === null) return false;
    this.currentSample = current;
    this.nextSample = next;
    this.phase = 0;
    this.hasPair = true;
    this.playing = true;
    return true;
  }

  renderSample() {
    if (!this.startIfBuffered() || !this.hasPair) return 0;
    const value = this.currentSample + (this.nextSample - this.currentSample) * this.phase;
    this.phase += this.sourceRate / sampleRate;
    while (this.phase >= 1) {
      this.currentSample = this.nextSample;
      const next = this.readSample();
      if (next === null) {
        this.playing = false;
        this.hasPair = false;
        this.phase = 0;
        return value;
      }
      this.nextSample = next;
      this.phase -= 1;
    }
    return value;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output) return true;
    const frames = output[0]?.length || 0;
    for (let frame = 0; frame < frames; frame += 1) {
      const value = this.renderSample();
      for (const channel of output) channel[frame] = value;
    }
    return true;
  }
}

registerProcessor("mulive-pcm-player", MulivePCMPlayer);
