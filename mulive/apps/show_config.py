"""Print the effective merged config for a catalog or a single app.

Usage:

  python -m mulive.apps.show_config                  # infra defaults + profile
  python -m mulive.apps.show_config --app spell      # merged app config
  MULIVE_PROFILE=aws python -m mulive.apps.show_config --app spell
  python -m mulive.apps.show_config --catalog muapps/apps.yml --app spell --sources

``--sources`` also lists which override files (apps.<profile>.yml,
config.<profile>.yml) actually matched a file on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mulive.apps.app_config import load_app_config

from .config_merge import active_profiles, load_bundle_layers, load_infra_layers

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "muapps" / "apps.yml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the effective merged Mulive config after profile overrides."
    )
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG),
        help="Path to muapps/apps.yml (default: %(default)s).",
    )
    parser.add_argument(
        "--app",
        default=None,
        help="Bundle name (folder under muapps/) to render. "
        "Omit to render just the infra defaults + profile overrides.",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Also list the override files that were loaded.",
    )
    return parser.parse_args(argv)


def render(args):
    catalog_path = Path(args.catalog).resolve()
    raw_catalog = load_app_config(catalog_path)
    schema = raw_catalog.get("schema")
    if schema not in (1, 2):
        raise SystemExit(f"Unsupported catalog schema: {schema!r}")

    profiles = active_profiles() if schema == 2 else ()
    sources: list[Path] = [catalog_path]

    if schema == 2:
        infra, profile_files = load_infra_layers(catalog_path, raw_catalog, profiles)
        sources.extend(profile_files)
    else:
        infra = {}

    if args.app is None:
        payload = {
            "schema": schema,
            "profiles": list(profiles),
            "infra": infra,
        }
    else:
        bundle_dir = catalog_path.parent / args.app
        if not bundle_dir.is_dir():
            raise SystemExit(f"No bundle directory: {bundle_dir}")
        manifest = load_app_config(bundle_dir / "app.yml")
        backend = manifest.get("backend") or {}
        config_rel = backend.get("config") or "config.yml"
        base_config = (bundle_dir / config_rel).resolve()
        if not base_config.is_file():
            raise SystemExit(f"No bundle config: {base_config}")
        sources.append(base_config)
        if schema == 2:
            merged, bundle_files = load_bundle_layers(
                bundle_dir, base_config, infra, profiles
            )
            sources.extend(bundle_files)
        else:
            merged = load_app_config(base_config)
        payload = {
            "schema": schema,
            "profiles": list(profiles),
            "app": args.app,
            "config": merged,
        }

    yaml.safe_dump(payload, sys.stdout, sort_keys=False, default_flow_style=False)
    if args.sources:
        print("# sources (in merge order):", file=sys.stdout)
        for source in sources:
            print(f"#   {source}", file=sys.stdout)


def main(argv=None):
    render(parse_args(argv))


if __name__ == "__main__":
    main()
