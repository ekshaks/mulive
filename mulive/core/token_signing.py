"""Shared URL-safe JSON and HMAC primitives for signed local tokens."""

import base64
import hashlib
import hmac
import json


def encode_json(value: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")


def decode_json(value: str) -> dict:
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    return json.loads(decoded)


def sign(secret: bytes, encoded: str) -> str:
    digest = hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def signature_matches(secret: bytes, encoded: str, supplied: str) -> bool:
    return hmac.compare_digest(sign(secret, encoded), supplied)
