"""Tests for TTS provider selection and Piper asset auto-fetch."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mulive.core.embed_stt_tts import TTSFlow
from mulive.core.tts_providers import KokoroOnnxTTSProvider, TTSConfig, create_tts_provider


class TestTtsProviderAliasResolution(unittest.TestCase):
    """Provider names are explicit; the default is in-process Kokoro."""

    def test_default_is_kokoro_onnx(self):
        self.assertEqual(TTSConfig().provider, "kokoro_onnx")

    def test_explicit_names_pass_through_unchanged(self):
        for name in ("kokoro_fastapi", "kokoro_onnx", "piper", "gemini"):
            self.assertEqual(TTSConfig(provider=name).provider, name)

    def test_kokoro_alias_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown TTS provider: kokoro"):
            TTSConfig(provider="kokoro")

    def test_factory_creates_default_provider(self):
        provider = create_tts_provider(TTSConfig(mode="local"))
        self.assertIsInstance(provider, KokoroOnnxTTSProvider)

    def test_embedded_flow_uses_default_provider(self):
        audio_output = object()
        with patch("mulive.core.embed_stt_tts.create_tts_provider") as factory:
            TTSFlow().create_speaker(audio_output)
        self.assertEqual(factory.call_args.args[0].provider, "kokoro_onnx")
        self.assertIs(factory.call_args.kwargs["audio_output"], audio_output)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown TTS provider: bogus"):
            TTSConfig(provider="bogus")


class TestPiperAssetAutoFetch(unittest.TestCase):
    """``_ensure_piper_asset`` should hit env override, cache, and download."""

    def test_env_override_returns_existing_file(self):
        from mulive.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "custom.onnx"
            model_path.write_bytes(b"stub model bytes")
            config_path = model_path.with_suffix(".onnx.json")
            config_path.write_text('{"sample_rate": 22050}')

            env = {
                "PIPER_MODEL_PATH": str(model_path),
                "PIPER_CONFIG_PATH": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(piper._ensure_piper_asset("model"), model_path)
                self.assertEqual(piper._ensure_piper_asset("config"), config_path)

    def test_env_override_raises_when_missing(self):
        from mulive.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.onnx"
            with patch.dict(os.environ, {"PIPER_MODEL_PATH": str(missing)}):
                with self.assertRaises(FileNotFoundError):
                    piper._ensure_piper_asset("model")

    def test_config_derived_from_model_override_directory(self):
        from mulive.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "voice.onnx"
            model_path.write_bytes(b"stub")
            config_path = model_path.with_suffix(".onnx.json")
            config_path.write_text('{"sample_rate": 22050}')
            env = {"PIPER_MODEL_PATH": str(model_path)}
            # Explicitly unset PIPER_CONFIG_PATH so the model-derived
            # fallback path is exercised.
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("PIPER_CONFIG_PATH", None)
                self.assertEqual(piper._ensure_piper_asset("config"), config_path)

    def test_cached_asset_is_used_when_present(self):
        from mulive.core.tts_providers import piper

        with TemporaryDirectory() as tmp:
            fake_home = Path(tmp)
            env = {"XDG_CACHE_HOME": str(fake_home)}
            with patch.dict(os.environ, env, clear=False):
                for k in ("PIPER_MODEL_PATH", "PIPER_CONFIG_PATH"):
                    os.environ.pop(k, None)
                cache_dir = fake_home / "mulive" / "piper"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_model = cache_dir / f"{piper.DEFAULT_PIPER_VOICE}.onnx"
                cached_model.write_bytes(b"cached-onnx")

                # Point the ext/piper bundled path at a nonexistent
                # directory to force falling through to the cache.
                with patch.object(
                    piper, "PROJECT_ROOT", Path(tmp) / "no-such-repo"
                ):
                    self.assertEqual(piper._ensure_piper_asset("model"), cached_model)

    def test_missing_asset_triggers_download(self):
        from mulive.core.tts_providers import piper

        downloads = []

        def _fake_download(url: str, dest: Path) -> Path:
            downloads.append((url, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded-bytes")
            return dest

        with TemporaryDirectory() as tmp:
            env = {"XDG_CACHE_HOME": str(tmp)}
            with patch.dict(os.environ, env, clear=False):
                for k in ("PIPER_MODEL_PATH", "PIPER_CONFIG_PATH"):
                    os.environ.pop(k, None)
                with patch.object(piper, "PROJECT_ROOT", Path(tmp) / "no-such-repo"):
                    with patch.object(piper, "_download_piper_asset", _fake_download):
                        result = piper._ensure_piper_asset("model")

                self.assertEqual(len(downloads), 1)
                url, dest = downloads[0]
                self.assertTrue(url.endswith(f"{piper.DEFAULT_PIPER_VOICE}.onnx"))
                self.assertEqual(result, dest)
                self.assertEqual(dest.read_bytes(), b"downloaded-bytes")


if __name__ == "__main__":
    unittest.main()
