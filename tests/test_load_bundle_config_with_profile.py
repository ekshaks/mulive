"""Tests for ``load_bundle_config_with_profile``.

Standalone entrypoints (``python -m muapps.<app>.app``) rely on this
helper to see the same profile-aware config as ``mulive.apps.run_apps``.
The bug it prevents is subtle: without it, per-app ``config.yml`` values
silently shadow ``apps.<profile>.yml`` overrides — which is exactly how
AWS deployments were still falling back to Kokoro-FastAPI.
"""

import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mulive.apps.app_config import load_bundle_config_with_profile


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"))


class TestLoadBundleConfigWithProfile(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write(
            self.root / "apps.yml",
            """
            schema: 2
            defaults:
              stt: { provider: mlx, variant: turbo, language: en }
              tts: { enabled: true, mode: browser, provider: kokoro_onnx }
            apps:
              - path: chess
                enabled: true
            """,
        )
        _write(
            self.root / "apps.aws.yml",
            """
            stt: { provider: faster_whisper, variant: base }
            tts: { provider: piper }
            """,
        )
        _write(
            self.root / "chess" / "config.yml",
            """
            app: { name: chess }
            stt: { variant: small }
            models: { text: groq:foo }
            """,
        )
        _write(
            self.root / "chess" / "config.aws.yml",
            """
            stt: { variant: base }
            """,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, path):
        with patch.dict(os.environ, {"MULIVE_APPS_CATALOG": str(self.root / "apps.yml")}):
            return load_bundle_config_with_profile(path)

    def test_local_profile_merges_defaults_and_per_app(self):
        with patch.dict(os.environ, {"MULIVE_PROFILE": "local"}):
            merged = self._config(self.root / "chess" / "config.yml")
        self.assertEqual(merged["tts"]["provider"], "kokoro_onnx")
        self.assertEqual(merged["stt"]["provider"], "mlx")
        self.assertEqual(merged["stt"]["variant"], "small")

    def test_aws_profile_overrides_per_app_via_config_aws_yml(self):
        with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
            merged = self._config(self.root / "chess" / "config.yml")
        self.assertEqual(merged["tts"]["provider"], "piper")
        self.assertEqual(merged["stt"]["provider"], "faster_whisper")
        # The per-app AWS override lifts variant back to base after the
        # per-app config.yml would have shadowed it with `small`.
        self.assertEqual(merged["stt"]["variant"], "base")

    def test_missing_catalog_falls_back_to_plain_load(self):
        # A config file outside any muapps tree loads as-is.
        with TemporaryDirectory() as tmp:
            standalone = Path(tmp) / "config.yml"
            _write(standalone, "app: { name: solo }\nfoo: 42\n")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MULIVE_PROFILE", None)
                os.environ.pop("MULIVE_APPS_CATALOG", None)
                merged = load_bundle_config_with_profile(standalone)
        self.assertEqual(merged, {"app": {"name": "solo"}, "foo": 42})

    def test_schema1_catalog_skips_profile_merge(self):
        # A schema-1 catalog signals legacy layout — no defaults, no profiles.
        _write(self.root / "apps.yml", "schema: 1\napps:\n  - {path: chess}\n")
        with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}):
            merged = self._config(self.root / "chess" / "config.yml")
        # Only the per-app config.yml applies — no AWS override reached it.
        self.assertNotIn("tts", merged)
        self.assertEqual(merged["stt"], {"variant": "small"})


if __name__ == "__main__":
    unittest.main()
