// app.js
// Main glue: UI <-> TextTransport

import { addMessage, getInputValue, clearInput } from "./ui.js";
import { TextTransport } from "./text_transport.js";

function initAppWS() {
  const form = document.getElementById("chat-form");

  // Create transport instance
  const transport = new TextTransport(`ws://${window.location.host}/ws`);
  transport.connect();

  // Form submit handler
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = getInputValue();
    if (!msg) return;

    // Show user message
    addMessage("user", msg);

    // Send to backend
    transport.sendMessage(msg);

    // Clear input
    clearInput();
  });

  // Incoming assistant message
  transport.onMessage((msg) => {
    addMessage("assistant", msg);
  });

  console.log("[App] Initialized");
}

document.addEventListener("DOMContentLoaded", initAppWS);
