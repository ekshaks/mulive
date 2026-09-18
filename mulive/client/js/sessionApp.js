import { createInteractions } from "./interaction.js";
import { createMediaControls } from "./mediaControls.js";
import { TrackManager } from "./trackManager.js";
import { createTranscriptView } from "./transcriptView.js";
import { initializeUserSelector } from "./userSelector.js";

const UI_API_VERSION = 2;

const elements = {
  appName: document.getElementById("appName"),
  appControls: document.getElementById("appControls"),
  appOverlay: document.getElementById("appOverlay"),
  audioButton: document.getElementById("muteAudioBtn"),
  celebration: document.getElementById("celebration"),
  connectionStatus: document.getElementById("connectionStatus"),
  appsLink: document.getElementById("appsLink"),
  localVideo: document.getElementById("localVideo"),
  minimizeVideoButton: document.getElementById("minimizeVideoBtn"),
  statusText: document.getElementById("statusText"),
  transcript: document.getElementById("transcriptionText"),
  userSelector: document.getElementById("userSelector"),
  cameraButton: document.getElementById("cameraToggleBtn"),
  videoPanel: document.getElementById("videoPanel"),
  videoState: document.getElementById("videoState"),
  workspace: document.querySelector(".workspace"),
};

const reloadButton = document.createElement("button");
reloadButton.className = "connection-reload";
reloadButton.type = "button";
reloadButton.textContent = "Reload";
reloadButton.hidden = true;
reloadButton.addEventListener("click", () => window.location.reload());
elements.connectionStatus.appendChild(reloadButton);

function setStatus(message, state = "connected") {
  elements.statusText.textContent = message;
  elements.connectionStatus.dataset.state = state;
  reloadButton.hidden = state !== "closed";
}

function handleConnectionState(state) {
  if (state === "connected") {
    setStatus("Connected", "connected");
    return;
  }
  if (state === "connecting") {
    setStatus("Connecting…", "connecting");
    return;
  }
  if (state === "disconnected" || state === "failed" || state === "closed") {
    setStatus("Connection closed. Please reload to continue.", "closed");
  }
}

function selectedAppId() {
  const match = window.location.pathname.match(/^\/apps\/([a-z][a-z0-9-]*)\/?$/);
  return match ? match[1] : null;
}

async function loadClientConfig(appId) {
  try {
    const response = await fetch(
      appId ? `/api/apps/${encodeURIComponent(appId)}` : "/client-config",
    );
    if (!response.ok) return {};
    const config = await response.json();
    if (typeof config.app_name === "string" && config.app_name.trim()) {
      elements.appName.textContent = config.app_name.trim();
      document.title = config.app_name.trim();
    }
    return config;
  } catch (error) {
    console.warn("Client configuration unavailable", error);
    return {};
  }
}

const transcriptView = createTranscriptView(elements.transcript);
const interactions = createInteractions(elements.celebration);

const messageHandlers = new Map();

function onServerEvent(type, handler) {
  if (typeof type !== "string" || typeof handler !== "function") {
    throw new TypeError("onServerEvent requires a message type and handler");
  }
  const handlers = messageHandlers.get(type) || [];
  handlers.push(handler);
  messageHandlers.set(type, handlers);
  return () => {
    const remaining = (messageHandlers.get(type) || []).filter(
      (candidate) => candidate !== handler,
    );
    messageHandlers.set(type, remaining);
  };
}

onServerEvent("transcript", (message) => transcriptView.append(message));
onServerEvent(
  "app_feedback",
  (message) => interactions.handleAppFeedback(message),
);

function loadStylesheet(url) {
  return new Promise((resolve) => {
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = url;
    stylesheet.onload = resolve;
    stylesheet.onerror = () => {
      console.error(`App stylesheet failed to load: ${url}`);
      resolve();
    };
    document.head.appendChild(stylesheet);
  });
}

async function loadAppUI(config) {
  if (config.ui_stylesheet) {
    await loadStylesheet(config.ui_stylesheet);
  }
  if (!config.ui_module) return;
  if (config.ui_api_version !== UI_API_VERSION) {
    console.error(
      `Unsupported app UI API version: ${config.ui_api_version}`,
    );
    return;
  }

  try {
    const appModule = await import(config.ui_module);
    if (typeof appModule.register !== "function") {
      throw new TypeError("App UI module must export register(context)");
    }
    await appModule.register(Object.freeze({
      appConfig: Object.freeze({ ...config }),
      slots: Object.freeze({
        controls: elements.appControls,
        overlay: elements.appOverlay,
      }),
      interactionRoot: elements.celebration,
      showCelebration: interactions.showCelebration,
      muteMicrophone: () => TrackManager.muteAudio(),
      onServerEvent,
      sendUICommand: (message) => TrackManager.sendData(message),
      uiApiVersion: UI_API_VERSION,
    }));
  } catch (error) {
    console.error("App UI module failed to register", error);
  }
}

function onDataMessage(rawData) {
  try {
    const message = JSON.parse(rawData);
    const type = message.type || "transcript";
    for (const handler of messageHandlers.get(type) || []) {
      try {
        handler(message);
      } catch (error) {
        console.error(`Client message handler failed for ${type}`, error);
      }
    }
  } catch (error) {
    console.error("Invalid data-channel message", error);
  }
}

createMediaControls({
  trackManager: TrackManager,
  elements,
  onStatus: setStatus,
});

(async () => {
  const appId = selectedAppId();
  if (appId) {
    elements.appsLink.hidden = false;
  }
  const [config, userId] = await Promise.all([
    loadClientConfig(appId),
    initializeUserSelector(elements.userSelector, { reloadOnChange: true }),
  ]);
  if (
    Array.isArray(config.capabilities) &&
    !config.capabilities.includes("video")
  ) {
    elements.videoPanel.hidden = true;
  }
  await loadAppUI(config);
  setStatus("Connecting…", "connecting");
  try {
    const offerPath = appId
      ? `/apps/${encodeURIComponent(appId)}/offer`
      : "/offer";
    const offerUrl = userId
      ? `${offerPath}?user_id=${encodeURIComponent(userId)}`
      : offerPath;
    await TrackManager.initConnection(onDataMessage, offerUrl, handleConnectionState);
    setStatus("Connected", "connected");
  } catch (error) {
    console.error("Connection failed", error);
    setStatus("Connection failed", "error");
  }
})();
