class MulivePCMWorklet extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const channel = inputs[0]?.[0];
    if (channel?.length) this.port.postMessage(channel.slice());
    for (const output of outputs) for (const channelOut of output) channelOut.fill(0);
    return true;
  }
}

registerProcessor("mulive-pcm", MulivePCMWorklet);
