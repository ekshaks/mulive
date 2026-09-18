export function createTranscriptView(container) {
  let hasMessages = false;

  function append({ role, content }) {
    if (!hasMessages) {
      container.textContent = "";
      hasMessages = true;
    }

    const message = document.createElement("div");
    message.className = `transcript-line transcript-${role}`;

    const roleLabel = document.createElement("span");
    roleLabel.className = "transcript-role";
    roleLabel.textContent = role === "user" ? "You" : "Assistant";

    const messageContent = document.createElement("span");
    messageContent.className = "transcript-content";
    messageContent.textContent = content;

    message.append(roleLabel, messageContent);
    container.appendChild(message);
    container.scrollTop = container.scrollHeight;
  }

  return { append };
}
