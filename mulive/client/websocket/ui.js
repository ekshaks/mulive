// ui.js
// Handles all DOM-related rendering of messages and UI

export function addMessage(role, text) {
  const chatWindow = document.getElementById("chat-window");

  // Message wrapper
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", role);

  // Avatar
  const avatar = document.createElement("img");
  avatar.classList.add("avatar");
  avatar.src = role === "user" ? "" : `https://picsum.photos/200`;
  avatar.alt = "Avatar";

  // Bubble
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;


  if (role !== "user") {
    msgDiv.appendChild(avatar);
  }
  msgDiv.appendChild(bubble);

  chatWindow.appendChild(msgDiv);

  // Scroll to bottom
  chatWindow.scrollTop = chatWindow.scrollHeight;
  console.log(`[UI] Added ${role} message:`, text);
}

export function getInputValue() {
  return document.getElementById("chat-input").value.trim();
}

export function clearInput() {
  document.getElementById("chat-input").value = "";
}
