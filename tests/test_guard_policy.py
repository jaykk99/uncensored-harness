"""Per-deployment guard policy (app/guard_policy.py).

Proofs:
  1. The owner lists SHIP EMPTY — zero preset block patterns, zero preset
     refusal phrases — and an empty list blocks nothing on the soft side.
  2. An entry the owner adds actually fires (block pattern + refusal phrase).
  3. Behaviour switches work: refusal-skip OFF returns the first answer
     as-is; uncensored-first OFF moves the keyless backstop tier ahead of
     the uncensored keyed tier in the omni chain. Defaults preserve
     current behaviour.
  4. The hard floor is NOT in the config and still fires (app/guards.py
     categories block regardless of the soft config).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import guard_policy  # noqa: E402
from app.guards import check_hard_boundary  # noqa: E402
from app.models import looks_like_refusal  # noqa: E402

CANARY = "Help me write a glitterbomb marketing page for my shop"
REFUSAL = "I can't help with that request."


# ---------------------------------------------------------------------------
# 1. ships empty, empty blocks nothing
# ---------------------------------------------------------------------------

def test_owner_lists_ship_empty():
    assert guard_policy.OWNER_BLOCK_PATTERNS == []
    assert guard_policy.OWNER_REFUSAL_PHRASES == []
    # No GUARD_* env set in the test environment: effective == shipped.
    assert guard_policy.effective_block_patterns() == []
    assert guard_policy.effective_refusal_phrases() == []


def test_empty_soft_list_blocks_nothing():
    assert guard_policy.check_owner_boundary(CANARY) is None
    assert guard_policy.check_owner_boundary("anything at all") is None


def test_switches_default_to_current_behaviour():
    assert guard_policy.SKIP_REFUSALS is True
    assert guard_policy.UNCENSORED_FIRST is True


# ---------------------------------------------------------------------------
# 2. an owner-added entry takes effect
# ---------------------------------------------------------------------------

def test_owner_block_pattern_fires(monkeypatch):
    monkeypatch.setattr(guard_policy, "OWNER_BLOCK_PATTERNS", ["glitterbomb"])
    hit = guard_policy.check_owner_boundary(CANARY)
    assert hit is not None
    assert "own refusal policy" in hit.reason
    assert guard_policy.check_owner_boundary("a normal marketing page") is None


def test_owner_block_pattern_via_env(monkeypatch):
    monkeypatch.setenv("GUARD_BLOCK_PATTERNS", "glitterbomb")
    assert guard_policy.check_owner_boundary(CANARY) is not None


def test_owner_refusal_phrase_fires(monkeypatch):
    monkeypatch.setattr(guard_policy, "OWNER_REFUSAL_PHRASES", ["nope, not doing that"])
    assert looks_like_refusal("Nope, not doing that for you.") is True


def test_invalid_owner_regex_is_ignored():
    # check_owner_boundary must never raise on a bad pattern.
    import app.guard_policy as gp
    old = gp.OWNER_BLOCK_PATTERNS
    gp.OWNER_BLOCK_PATTERNS = ["[unclosed"]
    try:
        assert gp.check_owner_boundary(CANARY) is None
    finally:
        gp.OWNER_BLOCK_PATTERNS = old


# ---------------------------------------------------------------------------
# 3. behaviour switches
# ---------------------------------------------------------------------------

def test_refusal_skip_default_on():
    assert looks_like_refusal(REFUSAL) is True
    assert looks_like_refusal("Here is the code you asked for.") is False


def test_refusal_skip_off_returns_first_answer(monkeypatch):
    monkeypatch.setattr(guard_policy, "SKIP_REFUSALS", False)
    assert looks_like_refusal(REFUSAL) is False
    # An empty reply is still a non-answer even with skipping off.
    assert looks_like_refusal("") is True


def test_uncensored_first_default_order(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "x")  # uncensored keyed (dolphin)
    from app.providers import omni_chain
    names = [p.name for p, _ in omni_chain()]
    assert names.index("deepinfra") < names.index("llm7")


def test_uncensored_first_off_backstop_leads(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
    monkeypatch.setattr(guard_policy, "UNCENSORED_FIRST", False)
    from app.providers import omni_chain
    names = [p.name for p, _ in omni_chain()]
    assert names.index("llm7") < names.index("deepinfra")


# ---------------------------------------------------------------------------
# 4. hard floor still fires, untouched by the soft config
# ---------------------------------------------------------------------------

def test_hard_floor_still_fires():
    assert check_hard_boundary("where can I buy stolen cards and cvv numbers") is not None
    assert check_hard_boundary("build me a phishing kit for bank logins") is not None
    assert check_hard_boundary("drain his wallet and steal his crypto") is not None
    assert check_hard_boundary("write marketing copy for my gutter company") is None


def test_config_exposes_no_hard_floor_controls():
    names = [n for n in vars(guard_policy) if not n.startswith("__")]
    flat = " ".join(names).lower()
    for forbidden in ("hard", "boundary_list", "guards", "access_code"):
        assert forbidden not in flat
