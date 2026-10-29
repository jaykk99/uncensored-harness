"""Per-deployment guard policy for THIS harness instance.

This app is a template: whoever deploys an instance sets their own refusal
policy here. Refusal policy is yours: edit this file (or the env vars below)
to set what your instance refuses; it ships empty.

WHAT LIVES HERE (soft policy, per-deployment):
  - OWNER_BLOCK_PATTERNS: regex sources matched against user messages.
    A match means THIS deployment refuses that message, with a plain
    "blocked by this deployment's own refusal policy" reply. Ships EMPTY —
    zero preset entries. Add your own, one regex per entry, or set
    GUARD_BLOCK_PATTERNS (newline-separated regex sources).
  - OWNER_REFUSAL_PHRASES: extra phrases that mark a MODEL reply as a
    refusal, so the router skips to the next provider. Ships EMPTY. Or set
    GUARD_REFUSAL_PHRASES (newline-separated plain phrases).
  - SKIP_REFUSALS (env GUARD_SKIP_REFUSALS, default ON): when ON, a model
    leg whose answer is refusal boilerplate counts as failed and the chain
    falls forward. When OFF, the first answer is returned as-is.
  - UNCENSORED_FIRST (env GUARD_UNCENSORED_FIRST, default ON): when ON,
    the omni chain tries uncensored keyed models before the lightly
    filtered keyless backstop. When OFF, the backstop tier leads.

WHAT DELIBERATELY DOES NOT LIVE HERE (hard floor, stays in code,
fail-closed, NOT configurable from this file):
  - app/guards.py — the hard-boundary pre-check: stolen cards, phishing /
    credential harvesting, draining or stealing funds, weapons of mass
    harm, CSAM. Those are the deployment-independent floor.
  - The access-code gate in app/main.py and the SSRF / tool-safety checks.
Nothing in this file can weaken, disable, or empty those.
"""

from __future__ import annotations

import os
import re

from .guards import BoundaryHit


def _bool_env(name: str, fallback: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return fallback


def _list_env(name: str) -> list[str]:
    raw = os.environ.get(name) or ""
    return [s.strip() for s in raw.splitlines() if s.strip()]


# Per-deployment refusal patterns for user messages (regex sources).
# SHIPS EMPTY.
OWNER_BLOCK_PATTERNS: list[str] = []

# Extra model-refusal phrases for the router's skip detection.
# SHIPS EMPTY.
OWNER_REFUSAL_PHRASES: list[str] = []

# Behaviour switches (defaults preserve current behaviour).
SKIP_REFUSALS: bool = _bool_env("GUARD_SKIP_REFUSALS", True)
UNCENSORED_FIRST: bool = _bool_env("GUARD_UNCENSORED_FIRST", True)


def effective_block_patterns() -> list[str]:
    """The deployment's block patterns: file entries plus env entries."""
    return [*OWNER_BLOCK_PATTERNS, *_list_env("GUARD_BLOCK_PATTERNS")]


def effective_refusal_phrases() -> list[str]:
    """The deployment's extra refusal phrases: file entries plus env."""
    return [*OWNER_REFUSAL_PHRASES, *_list_env("GUARD_REFUSAL_PHRASES")]


def check_owner_boundary(text: str) -> BoundaryHit | None:
    """Soft, per-deployment refusal check for user messages.

    Empty pattern list (the shipped default) blocks nothing. An
    owner-added pattern that matches means this deployment refuses the
    message. Invalid regex sources are ignored so a config typo can
    never crash the gate.
    """
    t = (text or "").lower()
    for source in effective_block_patterns():
        try:
            if re.search(source, t):
                return BoundaryHit(
                    reason="Blocked by this deployment's own refusal policy.",
                    alternative="This instance's owner set that policy in "
                                "app/guard_policy.py — edit it there if it "
                                "should be allowed here.",
                )
        except re.error:
            continue
    return None
