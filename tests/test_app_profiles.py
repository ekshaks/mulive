"""Tests for the profile-aware config merge + show_config CLI.

These are end-to-end tests: they build a temporary muapps-shaped tree,
load it via the real loader, and shell out to show_config to verify the
merged output.
"""

import io
import os
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from mulive.apps.config_merge import (
    active_profiles,
    deep_merge,
    merge_all,
)
from mulive.apps.loader import load_app_catalog
from mulive.apps.show_config import main as show_config_main


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"))


class _TempCatalog:
    """Build a minimal schema-2 catalog for one bundle in a temp dir."""

    def __init__(self, root: Path):
        self.root = root
        self.muapps = root / "muapps"
        self.muapps.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.muapps / "apps.yml"

    def write_catalog(self, body: str) -> None:
        _write(self.catalog_path, body)

    def write_profile(self, name: str, body: str) -> None:
        _write(self.muapps / f"apps.{name}.yml", body)

    def write_bundle(
        self,
        bundle: str,
        *,
        config: str,
        profile_configs: dict[str, str] | None = None,
    ) -> None:
        bundle_dir = self.muapps / bundle
        _write(
            bundle_dir / "app.yml",
            f"""
            schema: 1
            id: {bundle}
            title: {bundle.title()}
            description: {bundle} test bundle
            capabilities: [audio]
            backend:
              entrypoint: app:factory
              config: config.yml
            """,
        )
        _write(bundle_dir / "config.yml", config)
        _write(
            bundle_dir / "app.py",
            "def factory(*args, **kwargs):\n    return lambda session: None\n",
        )
        for name, text in (profile_configs or {}).items():
            _write(bundle_dir / f"config.{name}.yml", text)


class DeepMergeTests(unittest.TestCase):
    def test_scalar_and_list_override_but_dicts_recurse(self):
        base = {"a": 1, "b": {"x": 1, "y": [1, 2]}, "c": [1]}
        override = {"a": 2, "b": {"y": [9], "z": 3}, "c": [4, 5]}
        self.assertEqual(
            deep_merge(base, override),
            {"a": 2, "b": {"x": 1, "y": [9], "z": 3}, "c": [4, 5]},
        )

    def test_merge_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        deep_merge(base, override)
        self.assertEqual(base, {"a": {"b": 1}})
        self.assertEqual(override, {"a": {"c": 2}})

    def test_merge_all_skips_none(self):
        self.assertEqual(merge_all([{"a": 1}, None, {"a": 2}]), {"a": 2})


class ActiveProfilesTests(unittest.TestCase):
    def test_default_is_local(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MULIVE_PROFILE", None)
            self.assertEqual(active_profiles(), ("local",))

    def test_comma_separated_stack_preserves_order_dedup(self):
        with patch.dict(os.environ, {"MULIVE_PROFILE": "aws, staging ,aws"}):
            self.assertEqual(active_profiles(), ("aws", "staging"))

    def test_empty_env_var_falls_back_to_local(self):
        with patch.dict(os.environ, {"MULIVE_PROFILE": "  ,  "}):
            self.assertEqual(active_profiles(), ("local",))


class LoaderProfileTests(unittest.TestCase):
    def _build(self, tmp: str) -> _TempCatalog:
        catalog = _TempCatalog(Path(tmp))
        catalog.write_catalog(
            """
            schema: 2
            defaults:
              server: {https: true, host: 0.0.0.0, port: 9000}
              stt: {provider: mlx, model_size: turbo}
              tts: {enabled: true, provider: kokoro_onnx}
            apps:
              - path: demo
            """,
        )
        catalog.write_profile(
            "aws",
            """
            server: {https: false, max_concurrent_sessions: 2}
            stt:
              provider: faster_whisper
              model_size: base
              kwargs: {compute_type: int8, cpu_threads: 1}
            tts: {provider: kokoro_fastapi}
            """,
        )
        catalog.write_bundle(
            "demo",
            config="""
            app: {name: demo, mode: a}
            ocr: {debug: true}
            """,
            profile_configs={
                "aws": "server: {debug: true}\n",
            },
        )
        return catalog

    def test_local_profile_uses_defaults_only(self):
        with TemporaryDirectory() as tmp:
            catalog = self._build(tmp)
            with patch.dict(os.environ, {"MULIVE_PROFILE": "local"}):
                registry, merged_catalog = load_app_catalog(catalog.catalog_path)
            demo = registry.get("demo")
            self.assertEqual(demo.config["stt"]["provider"], "mlx")
            self.assertEqual(demo.config["stt"]["model_size"], "turbo")
            self.assertEqual(demo.config["tts"]["provider"], "kokoro_onnx")
            self.assertTrue(demo.config["server"]["https"])
            self.assertEqual(demo.config["app"]["name"], "demo")
            # Catalog-level merged infra is exposed at the top level.
            self.assertTrue(merged_catalog["server"]["https"])

    def test_aws_profile_merges_infra_and_bundle_overrides(self):
        with TemporaryDirectory() as tmp:
            catalog = self._build(tmp)
            with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
                registry, merged_catalog = load_app_catalog(catalog.catalog_path)
            demo = registry.get("demo")
            # apps.aws.yml overrode defaults:
            self.assertFalse(demo.config["server"]["https"])
            self.assertEqual(demo.config["server"]["max_concurrent_sessions"], 2)
            self.assertEqual(demo.config["stt"]["provider"], "faster_whisper")
            self.assertEqual(
                demo.config["stt"]["kwargs"],
                {"compute_type": "int8", "cpu_threads": 1},
            )
            self.assertEqual(demo.config["tts"]["provider"], "kokoro_fastapi")
            # config.aws.yml overrode mulive.debug on top of everything:
            self.assertTrue(demo.config["server"]["debug"])
            # App-specific keys still there:
            self.assertEqual(demo.config["ocr"]["debug"], True)
            # Merged infra promoted to catalog top level for run_apps:
            self.assertEqual(merged_catalog["server"]["max_concurrent_sessions"], 2)

    def test_schema_1_catalog_is_loaded_verbatim(self):
        with TemporaryDirectory() as tmp:
            catalog = _TempCatalog(Path(tmp))
            catalog.write_catalog(
                """
                schema: 1
                server: {https: true}
                apps:
                  - path: demo
                """,
            )
            # Even with a profile file present, schema 1 must ignore it.
            catalog.write_profile("aws", "server: {https: false}\n")
            catalog.write_bundle("demo", config="app: {name: demo}\n")
            with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
                registry, merged = load_app_catalog(catalog.catalog_path)
            demo = registry.get("demo")
            self.assertNotIn("server", demo.config)  # bundle config had no server
            self.assertTrue(merged["server"]["https"])  # from catalog top level


class ShowConfigTests(unittest.TestCase):
    def _build(self, tmp: str) -> _TempCatalog:
        catalog = _TempCatalog(Path(tmp))
        catalog.write_catalog(
            """
            schema: 2
            defaults:
              server: {https: true}
              stt: {provider: mlx}
            apps:
              - path: demo
            """,
        )
        catalog.write_profile("aws", "stt: {provider: faster_whisper}\n")
        catalog.write_bundle("demo", config="app: {name: demo}\n")
        return catalog

    def test_local_render(self):
        with TemporaryDirectory() as tmp:
            catalog = self._build(tmp)
            with patch.dict(os.environ, {"MULIVE_PROFILE": "local"}):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    show_config_main(
                        ["--catalog", str(catalog.catalog_path), "--app", "demo"]
                    )
            payload = yaml.safe_load(buf.getvalue())
            self.assertEqual(payload["profiles"], ["local"])
            self.assertEqual(payload["config"]["stt"]["provider"], "mlx")

    def test_aws_render_with_sources(self):
        with TemporaryDirectory() as tmp:
            catalog = self._build(tmp)
            with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    show_config_main(
                        [
                            "--catalog",
                            str(catalog.catalog_path),
                            "--app",
                            "demo",
                            "--sources",
                        ]
                    )
            text = buf.getvalue()
            # Split YAML payload from the trailing "# sources" comments.
            payload_text, _, sources_text = text.partition("# sources")
            payload = yaml.safe_load(payload_text)
            self.assertEqual(payload["profiles"], ["aws"])
            self.assertEqual(payload["config"]["stt"]["provider"], "faster_whisper")
            self.assertIn("apps.aws.yml", sources_text)
            self.assertIn("config.yml", sources_text)

    def test_infra_only_render(self):
        with TemporaryDirectory() as tmp:
            catalog = self._build(tmp)
            with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    show_config_main(["--catalog", str(catalog.catalog_path)])
            payload = yaml.safe_load(buf.getvalue())
            self.assertNotIn("app", payload)
            self.assertEqual(payload["infra"]["stt"]["provider"], "faster_whisper")


if __name__ == "__main__":
    unittest.main()
