"""Application-bundle configuration and UI-facing web configuration."""

from pathlib import Path

import yaml

from mulive.core.server_config import web_config

UI_API_VERSION = 2


def _app_asset_url(value):
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("App UI asset paths must be strings")
    if not value.startswith("/app-assets/") or ".." in value.split("/"):
        raise ValueError(f"App UI asset path must be under /app-assets/: {value}")
    return value


def load_app_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_bundle_config_with_profile(config_path):
    """Load a bundle config with shared defaults and active profile overrides."""
    from .config_merge import active_profiles, load_bundle_layers, load_infra_layers

    config_path = Path(config_path).resolve()
    bundle_dir = config_path.parent
    catalog_path = _find_apps_catalog(bundle_dir)
    if catalog_path is None:
        return load_app_config(config_path)

    raw_catalog = load_app_config(catalog_path)
    if raw_catalog.get("schema") != 2:
        return load_app_config(config_path)

    profiles = active_profiles()
    infra, _ = load_infra_layers(catalog_path, raw_catalog, profiles)
    merged, _ = load_bundle_layers(bundle_dir, config_path, infra, profiles)
    return merged


def _find_apps_catalog(start_dir):
    import os

    override = os.environ.get("MULIVE_APPS_CATALOG")
    if override:
        candidate = Path(override).resolve()
        return candidate if candidate.is_file() else None

    for parent in (start_dir, start_dir.parent, start_dir.parent.parent):
        candidate = parent / "apps.yml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def app_section(config, name, default=None):
    value = config.get(name)
    if value is None:
        return default or {}
    return value


def build_web_config(config, use_https=None):
    server = app_section(config, "server")
    app = app_section(config, "app")
    if use_https is None:
        use_https = mulive.get("https", True)
    overrides = {key: value for key, value in mulive.items() if key != "https"}
    overrides["client_config"] = {
        "app_name": app.get("display_name") or app.get("name") or "Mulive",
        "ui_api_version": UI_API_VERSION,
        "ui_module": _app_asset_url(app.get("ui_module")),
        "ui_stylesheet": _app_asset_url(app.get("ui_stylesheet")),
    }
    return web_config(use_https=use_https, **overrides)


def resolve_app_path(base_dir, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path
