"""In-memory fixed-window rate limiting for expensive endpoints.

Scoped per client IP. Deliberately process-local: fine for a single-worker
local deployment; a multi-instance deployment would swap this for a shared
store (e.g. Redis) behind the same dependency interface.
"""

import threading
import time

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request

from app.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.api.auth import AuthActor

_WINDOW_SECONDS = 60.0


class FixedWindowRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(self, key: str, limit: int) -> bool:
        """Record a hit for key; False when the current window is exhausted."""
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))
            if now - window_start >= _WINDOW_SECONDS:
                window_start, count = now, 0
            if count >= limit:
                return False
            self._windows[key] = (window_start, count + 1)
            return True


_limiter = FixedWindowRateLimiter()


def get_rate_limiter() -> FixedWindowRateLimiter:
    return _limiter


def enforce_rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
    limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
) -> None:
    actor: AuthActor | None = getattr(request.state, "actor", None)
    if actor is not None and actor.mode == "guest":
        _enforce_guest_limits(request, settings, limiter)
        return
    if settings.rate_limit_per_minute <= 0:
        return
    # Authenticated session users get their own bucket (one noisy user can't
    # exhaust everyone else's); API-key and local requests stay per-IP, which
    # keeps the pre-identity behavior byte-for-byte. `get_actor` stores the
    # resolved actor on request.state; when auth never ran, fall back to IP.
    if actor is not None and actor.mode == "session" and actor.user_id is not None:
        key = f"user:{actor.user_id}"
    else:
        key = request.client.host if request.client else "unknown"
    if not limiter.allow(key, settings.rate_limit_per_minute):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )


def _client_ip(request: Request) -> str:
    """Client IP; behind the reverse proxy, the first X-Forwarded-For hop.

    Uvicorn runs with --proxy-headers --forwarded-allow-ips=127.0.0.1 in
    deployment, so request.client is already the real client there; the
    header fallback covers proxies that are not configured that way.
    """
    host = request.client.host if request.client else "unknown"
    if host in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip() or host
    return host


def _enforce_guest_limits(
    request: Request, settings: Settings, limiter: FixedWindowRateLimiter
) -> None:
    """Guests: a per-IP limit AND a global cap shared by all guests.

    The global cap protects the server (every answer is CPU inference);
    the per-IP limit keeps one visitor from using all of it.
    """
    allowed = limiter.allow(
        f"guest:{_client_ip(request)}", settings.guest_rate_limit_per_minute
    ) and limiter.allow("guest:*", settings.guest_global_rate_limit_per_minute)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="The public demo is busy. Try again in a minute, or sign in.",
            headers={"Retry-After": str(int(_WINDOW_SECONDS))},
        )
