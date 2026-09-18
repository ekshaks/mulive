"""Worktree-independent locations for mutable Mulive runtime data."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Return the mutable runtime-data directory without creating it."""
    configured = os.environ.get("MULIVE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_data_home = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    return xdg_data_home / "mulive"


def users_path(catalog_path: Path, catalog: dict) -> Path:
    """Resolve users outside a worktree, falling back to a legacy catalog file."""
    configured = os.environ.get("MULIVE_USERS_PATH")
    if configured:
        return Path(configured).expanduser()

    shared = data_dir() / "users.yml"
    if shared.is_file():
        return shared

    legacy = catalog_path.parent / str(catalog.get("users") or "users.yml")
    return legacy if legacy.is_file() else shared


def user_database_path() -> Path:
    """Return the SQLite database path for shared user memory."""
    return data_dir() / "users.db"
