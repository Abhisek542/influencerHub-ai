"""Verifies the backend's JWT so each run stays tied to the user who started it.

The backend signs with the raw bytes of JWT_SECRET and picks HS256, HS384 or HS512 from the key length.
"""
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

_ALGORITHMS = {"HS256": hashlib.sha256, "HS384": hashlib.sha384, "HS512": hashlib.sha512}


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class User:
    email: str
    role: str


def _decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def verify_token(token: str, secret: str, now: float | None = None) -> User:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_decode(header_b64))
        digest = _ALGORITHMS[header["alg"]]
        expected = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), digest).digest()
        if not hmac.compare_digest(expected, _decode(signature_b64)):
            raise AuthError("bad signature")
        payload = json.loads(_decode(payload_b64))
        if payload["exp"] <= (time.time() if now is None else now):
            raise AuthError("expired")
        return User(email=payload["sub"], role=payload["role"])
    except AuthError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise AuthError("malformed token") from error
