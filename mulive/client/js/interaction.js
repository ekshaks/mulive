const CONFETTI_COLORS = ["#66e3c4", "#ffd166", "#ff7a90", "#8db8ff", "#f7a8ff"];

export function createInteractions(celebration) {
  let celebrationTimer;

  function showCelebration() {
    clearTimeout(celebrationTimer);
    celebration.querySelectorAll(".confetti").forEach((piece) => piece.remove());

    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      for (let index = 0; index < 28; index += 1) {
        const piece = document.createElement("span");
        piece.className = "confetti";
        piece.style.setProperty("--x", `${Math.random() * 100}%`);
        piece.style.setProperty("--color", CONFETTI_COLORS[index % CONFETTI_COLORS.length]);
        piece.style.setProperty("--duration", `${1.1 + Math.random() * 0.8}s`);
        piece.style.setProperty("--delay", `${Math.random() * 0.25}s`);
        piece.style.setProperty("--drift", `${-80 + Math.random() * 160}px`);
        celebration.appendChild(piece);
      }
    }

    celebration.hidden = false;
    celebrationTimer = setTimeout(() => {
      celebration.hidden = true;
    }, 1800);
  }

  function handleAppFeedback(message) {
    if (message.name === "answer_evaluated" && message.result === "correct") {
      showCelebration();
    }
  }

  return { handleAppFeedback, showCelebration };
}
