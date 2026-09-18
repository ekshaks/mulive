"""Identity verification and signed browser sessions for Mulive hosts.

The session layer is intentionally independent of the login method. The first
provider is a shared password; a future OAuth callback can issue the same
``AuthenticatedPrincipal`` session without changing protected routes.
"""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from getpass import getpass
from typing import Protocol
from urllib.parse import quote

from aiohttp import web

from .token_signing import decode_json, encode_json, sign, signature_matches


PASSWORD_HASH_ENV = "MULIVE_AUTH_PASSWORD_HASH"
COOKIE_SECRET_ENV = "MULIVE_AUTH_COOKIE_SECRET"
SESSION_DAYS_ENV = "MULIVE_AUTH_SESSION_DAYS"
SESSION_COOKIE = "mulive_session"
PBKDF2_ITERATIONS = 600_000
PRINCIPAL_KEY = web.RequestKey("principal", "AuthenticatedPrincipal")


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """A verified identity that may receive a browser session."""

    subject: str
    display_name: str


class IdentityProvider(Protocol):
    """Verifies a login credential and returns an authenticated identity."""

    def authenticate(self, credential: str) -> AuthenticatedPrincipal | None:
        ...


class SharedPasswordProvider:
    """Single-owner provider backed by a PBKDF2-SHA256 password hash."""

    def __init__(self, password_hash: str):
        self.password_hash = password_hash

    def authenticate(self, credential: str) -> AuthenticatedPrincipal | None:
        if verify_password(credential, self.password_hash):
            return AuthenticatedPrincipal(subject="owner", display_name="Owner")
        return None


class SignedSessionStore:
    """Issues and validates stateless, signed, expiring session cookies."""

    def __init__(self, secret: str, *, duration_seconds: int, secure: bool = True):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError(f"{COOKIE_SECRET_ENV} must be at least 32 bytes")
        if duration_seconds <= 0:
            raise ValueError("Session duration must be positive")
        self._secret = secret.encode("utf-8")
        self.duration_seconds = duration_seconds
        self.secure = secure

    def issue(self, principal: AuthenticatedPrincipal) -> str:
        payload = {
            "sub": principal.subject,
            "name": principal.display_name,
            "exp": int(time.time()) + self.duration_seconds,
        }
        encoded = encode_json(payload)
        return f"{encoded}.{sign(self._secret, encoded)}"

    def read(self, token: str | None) -> AuthenticatedPrincipal | None:
        if not token:
            return None
        try:
            encoded, supplied_signature = token.split(".", 1)
            if not signature_matches(self._secret, encoded, supplied_signature):
                return None
            payload = decode_json(encoded)
            if not isinstance(payload, dict) or int(payload["exp"]) < time.time():
                return None
            subject = payload["sub"]
            display_name = payload["name"]
            if not isinstance(subject, str) or not isinstance(display_name, str):
                return None
            return AuthenticatedPrincipal(subject=subject, display_name=display_name)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def set_cookie(self, response: web.StreamResponse, principal: AuthenticatedPrincipal) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            self.issue(principal),
            max_age=self.duration_seconds,
            httponly=True,
            secure=self.secure,
            samesite="Lax",
            path="/",
        )

    def clear_cookie(self, response: web.StreamResponse) -> None:
        response.del_cookie(SESSION_COOKIE, path="/")


class AppAuthentication:
    """Protects a Mulive host while delegating credential checks to a provider."""

    def __init__(
        self,
        provider: IdentityProvider,
        sessions: SignedSessionStore,
        *,
        login_html_path: str | os.PathLike,
    ):
        self.provider = provider
        self.sessions = sessions
        self.login_html_path = login_html_path

    @classmethod
    def from_environment(
        cls,
        *,
        login_html_path: str | os.PathLike,
        secure_cookie: bool,
    ) -> "AppAuthentication":
        password_hash = os.environ.get(PASSWORD_HASH_ENV)
        if not password_hash:
            raise RuntimeError(f"Missing required environment variable: {PASSWORD_HASH_ENV}")
        secret = os.environ.get(COOKIE_SECRET_ENV)
        if not secret:
            raise RuntimeError(f"Missing required environment variable: {COOKIE_SECRET_ENV}")
        days = int(os.environ.get(SESSION_DAYS_ENV, "7"))
        return cls(
            SharedPasswordProvider(password_hash),
            SignedSessionStore(secret, duration_seconds=days * 24 * 60 * 60, secure=secure_cookie),
            login_html_path=login_html_path,
        )

    @web.middleware
    async def middleware(self, request: web.Request, handler):
        if request.path not in {"/login", "/logout"}:
            principal = self.sessions.read(request.cookies.get(SESSION_COOKIE))
            if principal is None:
                if request.method in {"GET", "HEAD"} and not request.path.startswith("/api/"):
                    next_path = quote(request.path_qs, safe="/?=&")
                    raise web.HTTPFound(location=f"/login?next={next_path}")
                raise web.HTTPUnauthorized(text="Authentication required")
            request[PRINCIPAL_KEY] = principal
        return await handler(request)

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/login", self.login_page)
        app.router.add_post("/login", self.login)
        app.router.add_post("/logout", self.logout)

    async def login_page(self, request: web.Request):
        return web.FileResponse(self.login_html_path)

    async def login(self, request: web.Request):
        form = await request.post()
        password = form.get("password")
        principal = self.provider.authenticate(password) if isinstance(password, str) else None
        next_path = _safe_next(form.get("next") or request.query.get("next"))
        if principal is None:
            raise web.HTTPFound(location=f"/login?error=1&next={quote(next_path, safe='/?=&')}")
        response = web.HTTPFound(location=next_path)
        self.sessions.set_cookie(response, principal)
        raise response

    async def logout(self, request: web.Request):
        response = web.HTTPFound(location="/login")
        self.sessions.clear_cookie(response)
        raise response


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    if not password:
        raise ValueError("Password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, raw_iterations, raw_salt, expected_digest = password_hash.split("$", 3)
        iterations = int(raw_iterations)
        if algorithm != "pbkdf2_sha256" or iterations < 1:
            return False
        actual_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(raw_salt),
            iterations,
        ).hex()
        return hmac.compare_digest(actual_digest, expected_digest)
    except (TypeError, ValueError):
        return False


def generate_cookie_secret() -> str:
    return secrets.token_urlsafe(48)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _safe_next(value: object) -> str:
    if isinstance(value, str) and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mulive shared-password authentication settings.")
    parser.add_argument("command", choices=("generate",), nargs="?", default="generate")
    args = parser.parse_args()
    password = getpass("Shared app password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords did not match")
    print(f"export {PASSWORD_HASH_ENV}='{hash_password(password)}'")
    print(f"export {COOKIE_SECRET_ENV}='{generate_cookie_secret()}'")


if __name__ == "__main__":
    main()
