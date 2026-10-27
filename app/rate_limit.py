"""Per-IP failed-access-code rate limiter for app/main.py's _check_access.

Every HTTP endpoint behind HARNESS_ACCESS_CODE gates a high-value surface
(sandboxed shell/code execution, plus server-held GitHub/GitLab/Gmail/
Vercel/Fal tokens) with nothing stronger than a single shared string
compared with `!=`. Without this, there was no limit at all on how many
times a caller could guess that string over the network -- fine if the
owner sets a long random code, but the README/.env.example never say to,
and the HTTP comparison itself has no cost per guess.

In-memory, single-process: honest about not surviving a restart or
scaling past one instance, which matches this app's existing "personal
harness" scope (same trade-off the Vercel/Render gateways in this
portfolio make for their own rate limiters).
"""
from __future__ import annotations

import time

WINDOW_SECONDS = 10 * 60   # count failures within this sliding window
MAX_FAILURES = 10          # failures allowed inside the window before lockout
LOCKOUT_SECONDS = 15 * 60  # how long a locked-out IP stays locked out

_failures: dict[str, list[float]] = {}
_locked_until: dict[str, float] = {}


def is_locked(ip: str) -> float:
    """Seconds remaining locked out (0 if not locked)."""
    until = _locked_until.get(ip, 0.0)
    remaining = until - time.monotonic()
    return remaining if remaining > 0 else 0.0


def record_failure(ip: str) -> float:
    """Record a wrong-code attempt. Returns seconds locked out (0 if not
    locked yet by this attempt)."""
    now = time.monotonic()
    hits = [t for t in _failures.get(ip, []) if now - t < WINDOW_SECONDS]
    hits.append(now)
    if len(hits) >= MAX_FAILURES:
        _locked_until[ip] = now + LOCKOUT_SECONDS
        _failures[ip] = []
        return LOCKOUT_SECONDS
    _failures[ip] = hits
    return 0.0


def clear(ip: str) -> None:
    """A correct code resets this IP's failure history."""
    _failures.pop(ip, None)
    _locked_until.pop(ip, None)


def reset_all() -> None:
    """Test-only: wipe all state so one test's failures don't lock out the
    next (the suite reuses one client IP across many test functions)."""
    _failures.clear()
    _locked_until.clear()
