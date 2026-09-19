import tempfile
import unittest
from pathlib import Path

from mulive.apps.app_config import UI_API_VERSION, build_web_config
from mulive.server.server_asyncio import Server as AsyncServer
from mulive.server.server_fastapi_webrtc import Server as FastAPIServer


async def _unused_session(session):
    return None


class AppClientConfigTests(unittest.TestCase):
    def test_exposes_validated_app_ui_entry_points(self):
        config = build_web_config(
            {
                "app": {
                    "display_name": "External Game",
                    "ui_module": "/app-assets/index.js",
                    "ui_stylesheet": "/app-assets/style.css",
                },
                "server": {"https": False},
            }
        )

        self.assertEqual(
            config["client_config"],
            {
                "app_name": "External Game",
                "ui_api_version": UI_API_VERSION,
                "ui_module": "/app-assets/index.js",
                "ui_stylesheet": "/app-assets/style.css",
            },
        )

    def test_rejects_assets_outside_app_mount(self):
        with self.assertRaises(ValueError):
            build_web_config(
                {
                    "app": {"ui_module": "https://example.com/app.js"},
                    "server": {"https": False},
                }
            )

    def test_servers_register_external_asset_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            async_server = AsyncServer(_unused_session, app_assets_dir=assets)
            fastapi_server = FastAPIServer(_unused_session, app_assets_dir=assets)

            aiohttp_paths = {
                resource.get_info().get("prefix")
                for resource in async_server.app.router.resources()
            }
            self.assertIn("/app-assets", aiohttp_paths)
            self.assertTrue(
                any(route.path == "/app-assets" for route in fastapi_server.app.routes)
            )

    def test_missing_asset_directory_is_rejected(self):
        missing = Path(tempfile.gettempdir()) / "mulive-missing-app-assets"
        with self.assertRaises(ValueError):
            AsyncServer(_unused_session, app_assets_dir=missing)


if __name__ == "__main__":
    unittest.main()
