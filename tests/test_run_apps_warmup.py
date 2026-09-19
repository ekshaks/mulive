"""End-to-end tests for the run_apps warm-up scan.

These build a temporary muapps tree, load it through the real loader,
and inspect what the warm-up scan would preload — without actually
downloading models or importing torch/onnxruntime.
"""

import os
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mulive.apps.loader import load_app_catalog
from mulive.apps.run_apps import _collect_warm_up_targets, _warm_up_from_registry


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"))


def _make_bundle(
    root: Path,
    name: str,
    *,
    stt_provider: str = "faster_whisper",
    stt_model_size: str = "base",
    stt_kwargs: dict | None = None,
    tts_provider: str = "kokoro_onnx",
    tts_enabled: bool = True,
) -> None:
    stt_yaml = f"stt:\n  provider: {stt_provider}\n  model_size: {stt_model_size}\n"
    if stt_kwargs:
        pairs = "\n".join(f"    {k}: {v}" for k, v in stt_kwargs.items())
        stt_yaml += f"  kwargs:\n{pairs}\n"
    tts_yaml = (
        f"tts:\n  enabled: {'true' if tts_enabled else 'false'}\n"
        f"  provider: {tts_provider}\n"
        f"  mode: browser\n"
    )
    _write(
        root / name / "app.yml",
        f"""
        schema: 1
        id: {name}
        title: {name.title()}
        description: Test bundle.
        capabilities: [voice]
        backend:
          entrypoint: entry:make_runner
          config: config.yml
        """,
    )
    _write(root / name / "config.yml", stt_yaml + tts_yaml)
    _write(
        root / name / "entry.py",
        "def make_runner(config):\n    return lambda session: None\n",
    )


def _make_catalog(root: Path, aws_extra: str = "") -> Path:
    _write(
        root / "apps.yml",
        """
        schema: 2
        defaults:
          server: {https: false}
        apps:
          - {path: alpha}
          - {path: beta}
        """,
    )
    if aws_extra:
        _write(root / "apps.aws.yml", aws_extra)
    return root / "apps.yml"


class WarmUpScanTests(unittest.TestCase):
    def _load(self, tmp: Path):
        return load_app_catalog(_make_catalog(tmp))

    def test_collects_dedup_stt_and_tts_across_apps(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", stt_kwargs={"compute_type": "int8"})
            _make_bundle(root, "beta", stt_kwargs={"compute_type": "int8"})
            registry, _ = self._load(root)
            stt, tts = _collect_warm_up_targets(registry)
            self.assertEqual(
                stt,
                [("faster_whisper", "base", (("compute_type", "int8"),))],
            )
            self.assertEqual(tts, {"kokoro_onnx"})

    def test_keeps_distinct_stt_configs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", stt_model_size="base")
            _make_bundle(root, "beta", stt_model_size="small")
            registry, _ = self._load(root)
            stt, _ = _collect_warm_up_targets(registry)
            sizes = sorted(size for _, size, _ in stt)
            self.assertEqual(sizes, ["base", "small"])

    def test_skips_disabled_tts_and_non_streaming_kokoro_fastapi(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", tts_provider="kokoro_fastapi")
            _make_bundle(root, "beta", tts_enabled=False)
            registry, _ = self._load(root)
            _, tts = _collect_warm_up_targets(registry)
            self.assertEqual(tts, set())

    def test_picks_up_piper(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", tts_provider="piper")
            _make_bundle(root, "beta", tts_provider="piper")
            registry, _ = self._load(root)
            _, tts = _collect_warm_up_targets(registry)
            self.assertEqual(tts, {"piper"})

    def test_skips_mlx_stt(self):
        # MLX is Apple-Silicon-only and loads lazily inside WhisperSTT.__init__;
        # the warm-up scan must not try to preload it.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", stt_provider="mlx", stt_model_size="turbo")
            registry, _ = self._load(root)
            stt, _ = _collect_warm_up_targets(registry)
            self.assertEqual(stt, [])

    def test_warm_up_is_no_op_without_flag(self):
        # mulive.warm_up defaults to false: nothing should be imported.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(root, "alpha", stt_provider="faster_whisper", tts_provider="piper")
            registry, catalog = self._load(root)
            # Would raise if piper/faster_whisper imports were attempted.
            _warm_up_from_registry(registry, catalog)

    def test_warm_up_sets_silero_backend_env(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_bundle(
                root,
                "alpha",
                stt_provider="mlx",
                stt_model_size="turbo",
                tts_provider="kokoro_fastapi",
            )
            catalog_path = _make_catalog(
                root,
                aws_extra="server: {warm_up: true}\nsilero_backend: onnx\n",
            )
            with patch.dict(os.environ, {"MULIVE_PROFILE": "aws"}, clear=False):
                # Ensure a stale value from the shell doesn't hide the setdefault.
                os.environ.pop("SILERO_BACKEND", None)
                registry, catalog = load_app_catalog(catalog_path)
                with patch("mulive.core.turndet.warm_up_vad") as mock_vad:
                    _warm_up_from_registry(registry, catalog)
                self.assertEqual(os.environ.get("SILERO_BACKEND"), "onnx")
                mock_vad.assert_called_once()


if __name__ == "__main__":
    unittest.main()
