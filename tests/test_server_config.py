import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mulive.core.server_config import ssl_config, web_config


class ServerTLSConfigTests(unittest.TestCase):
    def test_uses_xdg_config_directory_by_default(self):
        with TemporaryDirectory() as tmp:
            cert_dir = Path(tmp) / "mulive" / "certs"
            cert_dir.mkdir(parents=True)
            keyfile = cert_dir / "key.pem"
            certfile = cert_dir / "cert.pem"
            keyfile.write_text("key")
            certfile.write_text("cert")

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp}, clear=True):
                config = ssl_config()

        self.assertEqual(config["ssl_keyfile"], str(keyfile))
        self.assertEqual(config["ssl_certfile"], str(certfile))

    def test_explicit_web_config_paths_override_environment(self):
        with TemporaryDirectory() as tmp:
            keyfile = Path(tmp) / "key.pem"
            certfile = Path(tmp) / "cert.pem"
            keyfile.write_text("key")
            certfile.write_text("cert")

            with patch.dict(
                os.environ,
                {
                    "MULIVE_SSL_KEYFILE": "/missing/env-key.pem",
                    "MULIVE_SSL_CERTFILE": "/missing/env-cert.pem",
                },
            ):
                config = web_config(
                    ssl_keyfile=str(keyfile),
                    ssl_certfile=str(certfile),
                )

        self.assertEqual(config["ssl_keyfile"], str(keyfile))
        self.assertEqual(config["ssl_certfile"], str(certfile))

    def test_missing_files_raise_actionable_error(self):
        with patch.dict(
            os.environ,
            {
                "MULIVE_SSL_KEYFILE": "/missing/key.pem",
                "MULIVE_SSL_CERTFILE": "/missing/cert.pem",
            },
        ):
            with self.assertRaisesRegex(FileNotFoundError, "use --http"):
                ssl_config()

    def test_http_ignores_tls_paths(self):
        config = web_config(
            use_https=False,
            ssl_keyfile="/missing/key.pem",
            ssl_certfile="/missing/cert.pem",
        )

        self.assertNotIn("ssl_keyfile", config)
        self.assertNotIn("ssl_certfile", config)


if __name__ == "__main__":
    unittest.main()
