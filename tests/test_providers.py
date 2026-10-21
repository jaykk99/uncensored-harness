"""Tests for app/providers.py — routing tables, chain ordering, model ids.

Chain order is the product's core promise (uncensored-first routing), so
the ordering invariants are pinned here. Env vars are monkeypatched; no
network is touched.
"""

from __future__ import annotations

import pytest

from app import providers
from app.providers import (
    ALL_PROVIDERS,
    KEYED_PROVIDERS,
    KEYLESS_PROVIDERS,
    get_provider,
    key_is_set,
    list_models,
    omni_chain,
    parse_model_id,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for p in ALL_PROVIDERS:
        if p.key_env:
            monkeypatch.delenv(p.key_env, raising=False)
    monkeypatch.delenv("HORDE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# parse_model_id / get_provider
# ---------------------------------------------------------------------------

def test_parse_known_prefix():
    assert parse_model_id("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")


def test_parse_unknown_prefix_returns_none():
    assert parse_model_id("nope:model-x") == (None, "nope:model-x")


def test_parse_bare_horde_model():
    mid = "koboldcpp/L3-8B-Stheno-v3.2"
    assert parse_model_id(mid) == (None, mid)


def test_parse_model_with_colons_after_prefix():
    assert parse_model_id("kilo:nvidia/nemotron:free") == ("kilo", "nvidia/nemotron:free")


def test_get_provider_known_and_unknown():
    assert get_provider("groq").label == "Groq"
    assert get_provider("nope") is None


def test_provider_names_unique():
    names = [p.name for p in ALL_PROVIDERS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# key_is_set
# ---------------------------------------------------------------------------

def test_keyless_always_live(clean_env):
    for p in KEYLESS_PROVIDERS:
        assert key_is_set(p) is True


def test_keyed_needs_env(monkeypatch):
    p = get_provider("groq")
    assert key_is_set(p) is False
    monkeypatch.setenv("GROQ_API_KEY", "  sk-test  ")
    assert key_is_set(p) is True


def test_whitespace_key_counts_as_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    assert key_is_set(get_provider("groq")) is False


# ---------------------------------------------------------------------------
# omni_chain ordering — the uncensored-first promise
# ---------------------------------------------------------------------------

def test_chain_empty_when_nothing_keyed():
    chain = omni_chain()
    # Only keyless providers contribute.
    assert chain, "keyless backstop must always be present"
    assert all(p is not None and p.keyless for p, _ in chain)


def test_keyless_backstop_always_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    names = [p.name for p, _ in omni_chain()]
    assert "llm7" in names and "kilo" in names


def test_uncensored_keyed_come_before_plain_keyed(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "x")   # uncensored (dolphin)
    monkeypatch.setenv("GROQ_API_KEY", "x")        # plain
    chain = omni_chain()
    names = [p.name for p, _ in chain]
    assert names.index("deepinfra") < names.index("groq")


def test_uncensored_keyed_come_before_keyless(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "x")    # uncensored (dolphin)
    chain = omni_chain()
    names = [p.name for p, _ in chain]
    assert names.index("together") < names.index("llm7")
    assert names.index("together") < names.index("kilo")


def test_explicit_pick_goes_first(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    chain = omni_chain("groq:openai/gpt-oss-120b")
    (p, m), *_ = chain
    assert p.name == "groq" and m == "openai/gpt-oss-120b"


def test_explicit_pick_without_key_is_ignored():
    chain = omni_chain("groq:openai/gpt-oss-120b")
    assert all(not (p is not None and p.name == "groq") for p, _ in chain)


def test_explicit_horde_pick_ignored_by_chain():
    # Horde is routed by models.chat's own legs, not the omni chain.
    chain = omni_chain("horde:koboldcpp/L3-8B-Stheno-v3.2")
    assert all(p is not None for p, _ in chain)


def test_chain_has_no_duplicates():
    chain = omni_chain()
    keys = [f"{p.name}/{m}" for p, m in chain]
    assert len(keys) == len(set(keys))


def test_chain_entries_reference_real_provider_models():
    by_name = {p.name: set(p.models) for p in ALL_PROVIDERS}
    for p, m in omni_chain():
        assert m in by_name[p.name]


# ---------------------------------------------------------------------------
# list_models — the /api/models + UI picker feed
# ---------------------------------------------------------------------------

def test_list_models_shape():
    ms = list_models()
    assert len(ms) > 20
    for m in ms:
        assert {"id", "label", "provider", "company",
                "keyless", "live", "uncensored"} <= set(m)
        prov, _, _model = m["id"].partition(":")
        assert prov == m["provider"]


def test_list_models_live_flags(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    by_prov: dict[str, list[dict]] = {}
    for m in list_models():
        by_prov.setdefault(m["provider"], []).append(m)
    assert all(m["live"] for m in by_prov["groq"])
    assert all(m["live"] for m in by_prov["llm7"])   # keyless
    cerebras = [m for m in by_prov.get("cerebras", [])]
    assert cerebras and all(not m["live"] for m in cerebras)


def test_horde_entries_marked_uncensored():
    horde = [m for m in list_models() if m["provider"] == "horde"]
    assert horde and all(m["uncensored"] and m["keyless"] for m in horde)


def test_pollinations_fallback_present():
    ids = [m["id"] for m in list_models()]
    assert "pollinations:openai" in ids


def test_never_routes_rlhf_refusers():
    joined = " ".join(m["id"] for m in list_models()).lower()
    assert "gpt-4" not in joined and "claude" not in joined and "gemini" not in joined
