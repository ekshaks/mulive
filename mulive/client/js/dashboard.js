import { initializeUserSelector } from "./userSelector.js";

const appList = document.getElementById("appList");
const userSelector = document.getElementById("userSelector");

function renderApps(apps) {
  appList.replaceChildren();

  if (!apps.length) {
    appList.innerHTML = '<p class="status">No apps are available.</p>';
    return;
  }

  for (const app of apps) {
    const card = document.createElement(app.available === false ? "article" : "a");
    card.className = "app-card";
    if (app.available === false) {
      card.classList.add("unavailable");
      card.setAttribute("aria-disabled", "true");
    } else {
      card.href = `/apps/${encodeURIComponent(app.id)}`;
    }

    const title = document.createElement("h2");
    title.textContent = app.title;

    const description = document.createElement("p");
    description.textContent = app.description;

    const capabilities = document.createElement("span");
    capabilities.className = "capabilities";
    capabilities.textContent =
      app.available === false
        ? "Unavailable"
        : (app.capabilities || []).join(" + ");

    card.append(title, description, capabilities);
    appList.append(card);
  }
}

async function loadApps() {
  try {
    const response = await fetch("/api/apps");
    if (!response.ok) throw new Error(`App list returned ${response.status}`);
    renderApps(await response.json());
  } catch (error) {
    console.error("Could not load apps", error);
    appList.innerHTML =
      '<p class="status error">Could not load the app list. Please refresh.</p>';
  }
}

initializeUserSelector(userSelector);
loadApps();
