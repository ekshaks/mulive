import os
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from mulive.core.auth import (
    AppAuthentication,
    AuthenticatedPrincipal,
    SharedPasswordProvider,
    SignedSessionStore,
    hash_password,
    verify_password,
)
from mulive.server.server_asyncio import Server


ROOT = Path(__file__).resolve().parents[2]


async def unused_session(session):
    await session.closed.wait()


class PasswordProviderTests(unittest.TestCase):
    def test_password_hash_round_trip_and_rejects_wrong_value(self):
        password_hash = hash_password("correct horse battery staple", iterations=1_000)

        self.assertTrue(verify_password("correct horse battery staple", password_hash))
        self.assertFalse(verify_password("wrong password", password_hash))
        self.assertFalse(verify_password("anything", "invalid"))

    def test_signed_session_rejects_tampering_and_expiry(self):
        sessions = SignedSessionStore("x" * 32, duration_seconds=60, secure=False)
        token = sessions.issue(AuthenticatedPrincipal("owner", "Owner"))

        self.assertEqual(sessions.read(token), AuthenticatedPrincipal("owner", "Owner"))
        self.assertIsNone(sessions.read(f"{token}x"))
        with patch("mulive.core.auth.time.time", return_value=10_000_000_000):
            self.assertIsNone(sessions.read(token))

    def test_environment_configuration_requires_both_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MULIVE_AUTH_PASSWORD_HASH"):
                AppAuthentication.from_environment(
                    login_html_path=ROOT / "client" / "login.html",
                    secure_cookie=True,
                )

        with patch.dict(os.environ, {"MULIVE_AUTH_PASSWORD_HASH": "value"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MULIVE_AUTH_COOKIE_SECRET"):
                AppAuthentication.from_environment(
                    login_html_path=ROOT / "client" / "login.html",
                    secure_cookie=True,
                )


class AuthenticationRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        password_hash = hash_password("test-password", iterations=1_000)
        authentication = AppAuthentication(
            SharedPasswordProvider(password_hash),
            SignedSessionStore("x" * 32, duration_seconds=3600, secure=False),
            login_html_path=ROOT / "client" / "login.html",
        )
        server = Server(run_session=unused_session, config={}, authentication=authentication)
        self.client = TestClient(TestServer(mulive.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_login_protects_pages_assets_apis_and_offers(self):
        page = await self.client.get("/", allow_redirects=False)
        self.assertEqual(page.status, 302)
        self.assertEqual(page.headers["Location"], "/login?next=/")

        asset = await self.client.get("/js/sessionApp.js", allow_redirects=False)
        self.assertEqual(asset.status, 302)

        api = await self.client.get("/api/apps", allow_redirects=False)
        self.assertEqual(api.status, 401)

        offer = await self.client.post("/offer", allow_redirects=False)
        self.assertEqual(offer.status, 401)

        login_page = await self.client.get("/login")
        self.assertEqual(login_page.status, 200)
        self.assertIn("Sign in", await login_page.text())

        rejected = await self.client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            allow_redirects=False,
        )
        self.assertEqual(rejected.status, 302)
        self.assertEqual(rejected.headers["Location"], "/login?error=1&next=/")

        accepted = await self.client.post(
            "/login",
            data={"password": "test-password", "next": "/"},
            allow_redirects=False,
        )
        self.assertEqual(accepted.status, 302)
        token = accepted.cookies["mulive_session"].value

        authenticated_page = await self.client.get(
            "/",
            headers={"Cookie": f"mulive_session={token}"},
        )
        self.assertEqual(authenticated_page.status, 200)

        logout = await self.client.post(
            "/logout",
            headers={"Cookie": f"mulive_session={token}"},
            allow_redirects=False,
        )
        self.assertEqual(logout.status, 302)
        self.assertIn("mulive_session=\"\"", logout.headers["Set-Cookie"])
