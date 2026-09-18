const SELECTED_USER_KEY = "mulive.selected_user";

export async function initializeUserSelector(
  container,
  { reloadOnChange = false } = {},
) {
  if (!container) return null;

  const label = document.createElement("label");
  label.className = "global-user-selector";

  const text = document.createElement("span");
  text.textContent = "User";

  const select = document.createElement("select");
  select.disabled = true;
  select.setAttribute("aria-label", "Select user");

  const loading = document.createElement("option");
  loading.textContent = "Loading…";
  select.appendChild(loading);
  label.append(text, select);
  container.replaceChildren(label);

  try {
    const response = await fetch("/api/users");
    if (!response.ok) throw new Error(`User list returned ${response.status}`);
    const data = await response.json();
    const users = Array.isArray(data.users) ? data.users : [];
    if (!users.length) throw new Error("No users are configured");

    const availableIds = new Set(users.map((user) => user.id));
    const storedUser = localStorage.getItem(SELECTED_USER_KEY);
    const selectedUser = availableIds.has(storedUser)
      ? storedUser
      : data.default_user;

    select.replaceChildren();
    for (const user of users) {
      const option = document.createElement("option");
      option.value = user.id;
      option.textContent = user.display_name || user.id;
      select.appendChild(option);
    }
    select.value = selectedUser;
    select.disabled = users.length < 2;
    localStorage.setItem(SELECTED_USER_KEY, selectedUser);

    select.addEventListener("change", () => {
      localStorage.setItem(SELECTED_USER_KEY, select.value);
      if (reloadOnChange) {
        window.location.reload();
      }
    });

    return selectedUser;
  } catch (error) {
    console.warn("User selector unavailable", error);
    container.hidden = true;
    return null;
  }
}
