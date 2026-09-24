from __future__ import annotations
import json
import time
from typing import Any, Optional

import requests

from config import settings

# key -> (value, expires_at_epoch_seconds_or_None)
_LOCAL_FALLBACK: dict[str, tuple[Any, Optional[float]]] = {}


def _using_upstash() -> bool:
    return bool(settings.redis_url and settings.redis_token)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.redis_token}"}


def cache_get(key: str) -> Optional[Any]:
    """Return the cached value for `key`, or None if missing/expired/on error."""
    if not _using_upstash():
        entry = _LOCAL_FALLBACK.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            _LOCAL_FALLBACK.pop(key, None)
            return None
        return value

    try:
        r = requests.get(f"{settings.redis_url}/get/{key}", headers=_headers(), timeout=5)
        r.raise_for_status()
        result = r.json().get("result")
        return json.loads(result) if result is not None else None
    except Exception as e:
        print(f"[cache] get('{key}') failed, treating as cache miss: {e}")
        return None


def cache_set(key: str, value: Any, ttl_seconds: int | None = None) -> bool:
    """Store `value` (must be JSON-serializable) under `key`."""
    if not _using_upstash():
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        _LOCAL_FALLBACK[key] = (value, expires_at)
        return True

    try:
        payload = json.dumps(value)
        r = requests.post(f"{settings.redis_url}/set/{key}", headers=_headers(),
                           data=payload, timeout=5)
        r.raise_for_status()
        if ttl_seconds:
            requests.post(f"{settings.redis_url}/expire/{key}/{ttl_seconds}",
                           headers=_headers(), timeout=5)
        return True
    except Exception as e:
        print(f"[cache] set('{key}') failed: {e}")
        return False


def cache_delete(key: str) -> bool:
    """Remove `key` from the cache. Safe to call even if it doesn't exist."""
    if not _using_upstash():
        _LOCAL_FALLBACK.pop(key, None)
        return True

    try:
        r = requests.post(f"{settings.redis_url}/del/{key}", headers=_headers(), timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[cache] delete('{key}') failed: {e}")
        return False


def rate_limit_check(identifier: str, limit: int, window_seconds: int) -> bool:
    """
    Fixed-window rate limiter. Returns True if the request is allowed,
    False if `identifier` (e.g. a user id or IP) has exceeded `limit`
    requests within the current `window_seconds` window.

    Fails OPEN on any Redis error — a caching outage should never itself
    block real users from using the app.
    """
    window_id = int(time.time() // window_seconds)
    key = f"ratelimit:{identifier}:{window_id}"

    if not _using_upstash():
        count = (_LOCAL_FALLBACK.get(key) or (0, None))[0] + 1
        _LOCAL_FALLBACK[key] = (count, time.time() + window_seconds)
        return count <= limit

    try:
        r = requests.post(f"{settings.redis_url}/incr/{key}", headers=_headers(), timeout=5)
        r.raise_for_status()
        count = r.json().get("result", 0)
        if count == 1:
            requests.post(f"{settings.redis_url}/expire/{key}/{window_seconds}",
                           headers=_headers(), timeout=5)
        return count <= limit
    except Exception as e:
        print(f"[cache] rate_limit_check failed, failing open: {e}")
        return True