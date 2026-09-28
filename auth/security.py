"""
auth/security.py — password hashing (Argon2id) and JWT creation/verification.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from config import settings

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def _secret() -> str:
    if not settings.jwt_secret_key:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. Generate one with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return settings.jwt_secret_key


def create_access_token(user_id: int, email: str, role: str) -> str:
    # Raises if no secret is configured — better to fail loudly at login than
    # to issue tokens signed with an empty (forgeable) key.
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "email": email, "role": role, "exp": expire}
    return jwt.encode(payload, _secret(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict | None:
    if not settings.jwt_secret_key:
        return None   # no secret configured: reject every token
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None