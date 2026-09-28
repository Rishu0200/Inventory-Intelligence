"""
api/rate_limit.py — per-user and global rate limiting for expensive endpoints.

Uses cache.redis_client.rate_limit_check: a fixed-window counter in Upstash
when configured, in-process otherwise (fine for Render's single worker).
It fails OPEN if Redis is down, so a cache outage never locks users out.
"""
from __future__ import annotations
from fastapi import Depends, HTTPException, status

from auth.dependencies import get_current_user
from cache.redis_client import rate_limit_check
from db.models import User


def rate_limit(scope: str, per_user: int, window_seconds: int = 60,
               global_limit: int | None = None):
    """
    Build a dependency that authenticates the user, then enforces:
      - `per_user` requests per window for each user
      - optionally `global_limit` requests per window across ALL users
        (protects the shared Groq quota)
    """
    def dependency(user: User = Depends(get_current_user)) -> User:
        # Per-user first, so one abusive user doesn't drain the global budget
        if not rate_limit_check(f"{scope}:user:{user.id}", per_user, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="You're sending requests too quickly. Please wait a minute and try again.",
                headers={"Retry-After": str(window_seconds)},
            )
        if global_limit is not None and not rate_limit_check(
            f"{scope}:global", global_limit, window_seconds
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The assistant is busy right now. Please try again in a minute.",
                headers={"Retry-After": str(window_seconds)},
            )
        return user

    return dependency