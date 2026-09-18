import importlib
import re
import sys
from pathlib import Path
from typing import Any

from mulive.apps.app_config import app_section, load_app_config

from .config_merge import active_profiles, load_bundle_layers, load_infra_layers
from .registry import APP_ID_RE, AppDefinition, AppRegistry


def load_app_catalog(path: Path) -> tuple[AppRegistry, dict[str, Any]]:
    """Load ``muapps/apps.yml`` and every app bundle it lists.

    Schema 1 (legacy): the catalog file is used as-is. No profile overrides
    and no ``defaults:`` block; per-bundle configs are loaded verbatim.

    Schema 2: the catalog's ``defaults:`` block plus any ``apps.<profile>.yml``
    files (selected by ``MULIVE_PROFILE``, default ``local``) form a shared
    infra layer that is deep-merged under every bundle's config. Per-bundle
    ``config.<profile>.yml`` files layer on top of the bundle's ``config.yml``.
    """
    catalog_path = Path(path).resolve()
    raw_catalog = load_app_config(catalog_path)
    schema = raw_catalog.get("schema")
    if schema not in (1, 2):
        raise ValueError("App catalog schema must be 1 or 2")

    entries = raw_catalog.get("apps")
    if not isinstance(entries, list):
        raise ValueError("App catalog apps must be a list")

    if schema == 2:
        profiles = active_profiles()
        infra, _ = load_infra_layers(catalog_path, raw_catalog, profiles)
        catalog = {**infra}
        catalog["schema"] = schema
        if raw_catalog.get("users") is not None:
            catalog["users"] = raw_catalog["users"]
        catalog["apps"] = entries
    else:
        profiles = ()
        infra = {}
        catalog = raw_catalog

    registry = AppRegistry()
    for index, entry in enumerate(entries):
        enabled = not isinstance(entry, dict) or bool(entry.get("enabled", True))
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Each catalog app must have a path")
            bundle_dir = _resolve_inside(
                catalog_path.parent,
                entry["path"],
                directory=True,
            )
            registry.register(
                load_app_bundle(
                    bundle_dir,
                    enabled=enabled,
                    infra=infra,
                    profiles=profiles,
                )
            )
        except Exception as exc:
            if not enabled:
                continue
            app_id, title, bundle_path = _unavailable_identity(
                catalog_path.parent,
                entry,
                index,
            )
            registry.record_unavailable(
                app_id=app_id,
                title=title,
                bundle_path=bundle_path,
                reason=f"{type(exc).__name__}: {exc}",
            )

    return registry, catalog


def load_app_bundle(
    bundle_dir: Path,
    *,
    enabled: bool = True,
    infra: dict[str, Any] | None = None,
    profiles: tuple[str, ...] = (),
) -> AppDefinition:
    """Load one app bundle, merging shared infra + per-profile overrides."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest = load_app_config(bundle_dir / "app.yml")
    if manifest.get("schema") != 1:
        raise ValueError(f"{bundle_dir}: app manifest schema must be 1")

    app_id = manifest.get("id")
    if not isinstance(app_id, str) or not APP_ID_RE.fullmatch(app_id):
        raise ValueError(f"{bundle_dir}: invalid app id")
    title = _required_text(manifest, "title", bundle_dir)
    description = _required_text(manifest, "description", bundle_dir)

    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(item, str) or not item for item in capabilities)
    ):
        raise ValueError(f"{bundle_dir}: capabilities must be non-empty strings")

    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        raise ValueError(f"{bundle_dir}: backend must be an object")
    entrypoint = _required_text(backend, "entrypoint", bundle_dir)
    config_path = _resolve_inside(
        bundle_dir,
        _required_text(backend, "config", bundle_dir),
    )
    if infra or profiles:
        config, _ = load_bundle_layers(bundle_dir, config_path, infra or {}, profiles)
    else:
        config = load_app_config(config_path)
    session_runner_factory = _load_entrypoint(bundle_dir, entrypoint)

    ui = manifest.get("ui") or {}
    if not isinstance(ui, dict):
        raise ValueError(f"{bundle_dir}: ui must be an object")
    module_path = _optional_bundle_web_file(bundle_dir, ui.get("module"))
    stylesheet_path = _optional_bundle_web_file(bundle_dir, ui.get("stylesheet"))
    assets_dir = bundle_dir / "web" if module_path or stylesheet_path else None

    return AppDefinition(
        id=app_id,
        title=title,
        description=description,
        capabilities=tuple(dict.fromkeys(capabilities)),
        bundle_dir=bundle_dir,
        assets_dir=assets_dir,
        ui_module=_asset_url(app_id, assets_dir, module_path),
        ui_stylesheet=_asset_url(app_id, assets_dir, stylesheet_path),
        transport_config=dict(app_section(config, "server")),
        config=config,
        session_runner_factory=session_runner_factory,
        enabled=enabled,
    )


def _load_entrypoint(bundle_dir: Path, entrypoint: str):
    try:
        module_name, attribute = entrypoint.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"{bundle_dir}: invalid backend entrypoint") from exc
    if not module_name or not attribute or "." in module_name:
        raise ValueError(f"{bundle_dir}: entrypoint must be bundle-relative")

    parent = str(bundle_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{bundle_dir.name}.{module_name}")
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"{bundle_dir}: backend entrypoint is not callable")
    return factory


def _required_text(mapping: dict, key: str, owner: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}: {key} must be a non-empty string")
    return value.strip()


def _resolve_inside(base: Path, relative: str, *, directory: bool = False) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Bundle paths must be relative: {relative}")
    resolved = (base / candidate).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Bundle path escapes its owner: {relative}")
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        raise ValueError(f"Bundle path does not exist: {resolved}")
    return resolved


def _optional_bundle_web_file(bundle_dir: Path, relative: Any) -> Path | None:
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative.startswith("web/"):
        raise ValueError(f"{bundle_dir}: UI assets must be under web/")
    return _resolve_inside(bundle_dir, relative)


def _asset_url(
    app_id: str,
    assets_dir: Path | None,
    path: Path | None,
) -> str | None:
    if path is None:
        return None
    relative = path.relative_to(assets_dir).as_posix()
    return f"/app-assets/{app_id}/{relative}"


def _unavailable_identity(
    catalog_dir: Path,
    entry: Any,
    index: int,
) -> tuple[str, str, str]:
    raw_path = entry.get("path") if isinstance(entry, dict) else None
    raw_id = entry.get("id") if isinstance(entry, dict) else None
    raw_title = entry.get("title") if isinstance(entry, dict) else None

    manifest = {}
    if isinstance(raw_path, str):
        candidate = Path(raw_path)
        resolved = (catalog_dir / candidate).resolve()
        if (
            not candidate.is_absolute()
            and resolved.is_relative_to(catalog_dir.resolve())
            and (resolved / "app.yml").is_file()
        ):
            try:
                manifest = load_app_config(resolved / "app.yml")
            except Exception:
                manifest = {}

    candidate_id = raw_id or manifest.get("id")
    if not isinstance(candidate_id, str) or not APP_ID_RE.fullmatch(candidate_id):
        basename = Path(raw_path).name if isinstance(raw_path, str) else ""
        candidate_id = re.sub(r"[^a-z0-9-]+", "-", basename.lower()).strip("-")
        if not candidate_id or not candidate_id[0].isalpha():
            candidate_id = f"app-{index + 1}"

    candidate_title = raw_title or manifest.get("title")
    if not isinstance(candidate_title, str) or not candidate_title.strip():
        candidate_title = candidate_id.replace("-", " ").title()

    return candidate_id, candidate_title.strip(), str(raw_path or "")
