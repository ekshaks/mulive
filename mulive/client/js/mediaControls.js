function setControl(button, { pressed, text, label, state = "", icon }) {
  button.setAttribute("aria-pressed", String(pressed));
  button.setAttribute("aria-label", label);
  button.querySelector(".btn-text").textContent = text;
  button.dataset.state = state;
  if (icon) {
    button.querySelector(".control-icon use")?.setAttribute("href", `#${icon}`);
  }
}

function permissionMessage(kind, error) {
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return `${kind} permission denied`;
  }
  return `${kind} unavailable`;
}

export function createMediaControls({ trackManager, elements, onStatus }) {
  const state = {
    audioEnabled: false,
    videoOn: false,
    videoMinimized: false,
  };

  const {
    localVideo,
    workspace,
    videoPanel,
    videoState,
    minimizeVideoButton,
    cameraButton,
    audioButton,
  } = elements;

  minimizeVideoButton.addEventListener("click", () => {
    state.videoMinimized = !state.videoMinimized;
    videoPanel.dataset.minimized = String(state.videoMinimized);
    workspace.classList.toggle("video-minimized", state.videoMinimized);
    minimizeVideoButton.setAttribute("aria-expanded", String(!state.videoMinimized));
    minimizeVideoButton.setAttribute(
      "aria-label",
      state.videoMinimized ? "Restore camera preview" : "Minimize camera preview",
    );
  });

  cameraButton.addEventListener("click", async () => {
    cameraButton.disabled = true;
    setControl(cameraButton, {
      pressed: state.videoOn,
      text: state.videoOn ? "Stopping…" : "Starting…",
      label: state.videoOn ? "Stopping camera" : "Starting camera",
      state: "connecting",
      icon: state.videoOn ? "icon-camera-on" : "icon-camera-off",
    });
    videoState.textContent = state.videoOn ? "Stopping camera…" : "Requesting camera…";

    try {
      if (!state.videoOn) {
        await trackManager.startVideo(localVideo);
        state.videoOn = true;
      } else {
        await trackManager.stopVideo(localVideo);
        state.videoOn = false;
      }

      videoPanel.dataset.video = state.videoOn ? "on" : "off";
      videoState.textContent = state.videoOn ? "Video on" : "Video off";
      setControl(cameraButton, state.videoOn
        ? { pressed: true, text: "Camera on", label: "Turn camera off", icon: "icon-camera-on" }
        : { pressed: false, text: "Camera off", label: "Turn camera on", icon: "icon-camera-off" });
    } catch (error) {
      console.error("Video toggle error", error);
      const message = permissionMessage("Camera", error);
      videoState.textContent = message;
      onStatus(message, "error");
      setControl(cameraButton, {
        pressed: false,
        text: "Camera blocked",
        label: `${message}. Try camera again`,
        state: "denied",
        icon: "icon-camera-off",
      });
    } finally {
      cameraButton.disabled = false;
    }
  });

  audioButton.addEventListener("click", async () => {
    audioButton.disabled = true;
    setControl(audioButton, {
      pressed: state.audioEnabled,
      text: state.audioEnabled ? "Muting…" : "Starting…",
      label: state.audioEnabled ? "Muting microphone" : "Starting microphone",
      state: "connecting",
      icon: state.audioEnabled ? "icon-mic-on" : "icon-mic-off",
    });

    try {
      if (!trackManager.audioTrack) {
        await trackManager.initAudio();
        trackManager.unmuteAudio();
        state.audioEnabled = true;
      } else if (state.audioEnabled) {
        trackManager.muteAudio();
        state.audioEnabled = false;
      } else {
        trackManager.unmuteAudio();
        state.audioEnabled = true;
      }

      setControl(audioButton, state.audioEnabled
        ? { pressed: true, text: "Mic on", label: "Mute microphone", icon: "icon-mic-on" }
        : { pressed: false, text: "Mic off", label: "Turn microphone on", icon: "icon-mic-off" });
    } catch (error) {
      console.error("Audio toggle failed", error);
      const message = permissionMessage("Microphone", error);
      onStatus(message, "error");
      setControl(audioButton, {
        pressed: false,
        text: "Mic blocked",
        label: `${message}. Try microphone again`,
        state: "denied",
        icon: "icon-mic-off",
      });
    } finally {
      audioButton.disabled = false;
    }
  });

  return { state };
}
