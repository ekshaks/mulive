"""Profile-aware config loading with 3-layer deep-merge.

Layer order (later wins):

  1. ``muapps/apps.yml`` -> ``defaults:`` block (shared infra for every profile)
  2. ``muapps/apps.<profile>.yml`` (per-environment infra overrides)
  3. ``<bundle>/config.yml`` (app-specific base config)
  4. ``<bundle>/config.<profile>.yml`` (per-environment app overrides)

Layers 2 and 4 stack when ``MULIVE_PROFILE`` is comma-separated
(e.g. ``MULIVE_PROFILE=aws,staging`` applies aws then staging).

Deep-merge rule: dicts recurse; lists and scalars replace.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mulive.apps.app_config import load_app_config

DEFAULT_PROFILE = "local"


def active_profiles() -> tuple[str, ...]:
    """Return the ordered profile stack from ``MULIVE_PROFILE``.

    Empty or unset ``MULIVE_PROFILE`` yields ``("local",)``. Whitespace and
    duplicates are removed while preserving order.
    """
    raw = os.environ.get("MULIVE_PROFILE", DEFAULT_PROFILE)
    seen: dict[str, None] = {}
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            seen[name] = None
    return tuple(seen) or (DEFAULT_PROFILE,)


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` into ``base``.

    Dicts merge key-by-key; every other type replaces. Neither input is
    mutated; a new nested structure is returned.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = _clone(value)
        return merged
    return _clone(override)


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def merge_all(layers: Iterable[Any]) -> dict:
    """Fold ``deep_merge`` over an iterable of layers starting from ``{}``."""
    result: Any = {}
    for layer in layers:
        if layer is None:
            continue
        result = deep_merge(result, layer)
    if not isinstance(result, dict):
        raise ValueError("Merged config root must be a mapping")
    return result


def load_infra_layers(
    catalog_path: Path,
    raw_catalog: dict,
    profiles: Iterable[str],
) -> tuple[dict, list[Path]]:
    """Collect infra layers: ``defaults:`` + each ``apps.<profile>.yml``.

    Returns the merged infra dict and the list of override files that were
    actually loaded (so callers can report which profiles matched).
    """
    layers: list[Any] = [raw_catalog.get("defaults") or {}]
    loaded: list[Path] = []
    for profile in profiles:
        override = catalog_path.parent / f"apps.{profile}.yml"
        if override.is_file():
            layers.append(load_app_config(override))
            loaded.append(override)
    return merge_all(layers), loaded


def load_bundle_layers(
    bundle_dir: Path,
    base_config_path: Path,
    infra: dict,
    profiles: Iterable[str],
) -> tuple[dict, list[Path]]:
    """Merge infra + bundle base config + each ``config.<profile>.yml``.

    ``base_config_path`` is the file named by the bundle's ``backend.config``
    manifest entry. Per-profile files live next to it in the same bundle.
    """
    layers: list[Any] = [infra, load_app_config(base_config_path)]
    loaded: list[Path] = []
    for profile in profiles:
        override = bundle_dir / f"config.{profile}.yml"
        if override.is_file():
            layers.append(load_app_config(override))
            loaded.append(override)
    return merge_all(layers), loaded
