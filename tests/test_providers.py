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
# Health-aware ordering — poor legs demoted within their tier, never across
# ---------------------------------------------------------------------------

def _row(provider, model, attempts=10, rate=1.0, stale=False):
    return {"provider": provider, "model": model, "attempts": attempts,
            "success_rate": rate, "stale": stale}


def test_health_order_demotes_failing_leg_within_tier():
    stats = [_row("llm7", "codestral-latest", rate=0.0)]
    chain = omni_chain(stats=stats)
    keys = [f"{p.name}/{m}" for p, m in chain]
    bad = "llm7/codestral-latest"
    assert keys.index(bad) > keys.index("kilo/kilo-auto/free")
    # Still ahead of nothing keyed: no keys set, so the chain ends at tier 2.
    assert keys[-1] == bad


def test_health_order_stale_leg_demoted():
    stats = [_row("llm7", "codestral-latest", stale=True)]
    chain = omni_chain(stats=stats)
    keys = [f"{p.name}/{m}" for p, m in chain]
    assert keys[-1] == "llm7/codestral-latest"


def test_health_order_never_crosses_tiers(monkeypatch):
    # Even a failing uncensored-keyed leg outranks keyless legs: the
    # uncensored-first promise holds regardless of stats.
    monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
    stats = [_row("deepinfra",
                  "cognitivecomputations/dolphin-2.9.1-llama-3-70b", rate=0.0)]
    chain = omni_chain(stats=stats)
    names = [p.name for p, _ in chain]
    assert names.index("deepinfra") < names.index("llm7")
    assert names.index("deepinfra") < names.index("kilo")


def test_health_order_good_leg_keeps_static_place():
    stats = [_row("llm7", "codestral-latest", rate=1.0)]
    chain = omni_chain(stats=stats)
    (p, m), *_ = chain
    assert p.name == "llm7" and m == "codestral-latest"


def test_health_order_ignores_thin_data():
    # One failure is noise, not a pattern — no demotion.
    stats = [_row("llm7", "codestral-latest", attempts=1, rate=0.0)]
    chain = omni_chain(stats=stats)
    (p, m), *_ = chain
    assert p.name == "llm7" and m == "codestral-latest"


def test_health_order_ignores_unknown_legs():
    stats = [_row("nope", "ghost-model", rate=0.0)]
    chain = omni_chain(stats=stats)
    assert chain == omni_chain()


def test_health_order_disabled_via_env(monkeypatch):
    monkeypatch.setenv("ROUTER_HEALTH_ORDER", "0")
    stats = [_row("llm7", "codestral-latest", rate=0.0)]
    assert omni_chain(stats=stats) == omni_chain()


def test_health_order_stable_for_equal_penalty():
    # Good legs never shuffle: penalty-0 entries keep static relative order.
    stats = [_row("kilo", "kilo-auto/free", rate=0.0)]
    chain = omni_chain(stats=stats)
    static = omni_chain()
    good_static = [k for k in (f"{p.name}/{m}" for p, m in static)
                   if k != "kilo/kilo-auto/free"]
    good_health = [k for k in (f"{p.name}/{m}" for p, m in chain)
                   if k != "kilo/kilo-auto/free"]
    assert good_health == good_static


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


def test_list_models_skips_horde_when_import_fails(monkeypatch):
    # list_models guards the Horde import so a broken models module
    # still lists the other providers instead of crashing the picker.
    from app import models as models_mod
    monkeypatch.delattr(models_mod, "HORDE_UNCENSORED_MODELS")
    ms = list_models()
    assert not any(m["provider"] == "horde" for m in ms)
    assert any(m["provider"] == "llm7" for m in ms)


def test_pollinations_fallback_present():
    ids = [m["id"] for m in list_models()]
    assert "pollinations:openai" in ids


def test_never_routes_rlhf_refusers():
    joined = " ".join(m["id"] for m in list_models()).lower()
    assert "gpt-4" not in joined and "claude" not in joined and "gemini" not in joined


# ---------------------------------------------------------------------------
# HTTP call paths — stubbed httpx transport, no network
# ---------------------------------------------------------------------------

import json as _json

import httpx


def _mock_client(monkeypatch, module, handler):
    """Replace module._client with a factory returning a MockTransport client."""
    def factory(timeout):
        return httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module, "_client", factory)


def _ok_message(content="hello", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def test_call_openai_success(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=_ok_message("hi there")))
    p = get_provider("groq")
    msg = providers.call_openai_compatible(
        p, "openai/gpt-oss-120b", [{"role": "user", "content": "hi"}])
    assert msg["content"] == "hi there"


def test_call_openai_sends_key_when_set(monkeypatch):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("Authorization")
        seen["body"] = _json.loads(req.content)
        return httpx.Response(200, json=_ok_message("ok"))
    _mock_client(monkeypatch, providers, handler)
    monkeypatch.setenv("GROQ_API_KEY", "sk-abc")
    providers.call_openai_compatible(
        get_provider("groq"), "openai/gpt-oss-120b",
        [{"role": "user", "content": "hi"}], max_tokens=100)
    assert seen["auth"] == "Bearer sk-abc"
    # max_tokens floors at 700.
    assert seen["body"]["max_tokens"] == 700
    assert seen["body"]["model"] == "openai/gpt-oss-120b"


def test_call_openai_keyless_sends_no_auth(monkeypatch):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, json=_ok_message("ok"))
    _mock_client(monkeypatch, providers, handler)
    providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}])
    assert seen["auth"] is None


def test_call_openai_tools_in_body(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = _json.loads(req.content)
        return httpx.Response(200, json=_ok_message("ok"))
    _mock_client(monkeypatch, providers, handler)
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}], tools=tools)
    assert seen["body"]["tools"] == tools
    assert seen["body"]["tool_choice"] == "auto"


def test_call_openai_reasoning_fallback(monkeypatch):
    # Reasoning-only Kilo models put text in `reasoning`, content null.
    body = {"choices": [{"message": {"role": "assistant", "content": None,
                                     "reasoning": "thought out loud"}}]}
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=body))
    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}])
    assert msg["content"] == "thought out loud"


def test_call_openai_empty_choices_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(RuntimeError, match="empty choices"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}])


def test_call_openai_empty_response_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=_ok_message("")))
    with pytest.raises(RuntimeError, match="empty response"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}])


def test_call_openai_empty_response_with_tool_calls_ok(monkeypatch):
    tc = [{"id": "c1", "type": "function",
           "function": {"name": "web_search", "arguments": "{}"}}]
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=_ok_message("", tc)))
    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}])
    assert msg["tool_calls"] == tc


def test_call_openai_retryable_status_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(429, text="slow down"))
    with pytest.raises(RuntimeError, match="kilo HTTP 429"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}])


def test_call_openai_400_includes_body_snippet(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(400, text="bad model name"))
    with pytest.raises(RuntimeError, match="bad model name"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}])


def test_call_openai_streaming_via_on_token(monkeypatch):
    sse = ('data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, content=sse.encode()))
    seen = []
    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}], on_token=seen.append)
    assert msg["content"] == "hello"
    assert seen == ["hel", "lo"]


def test_call_openai_streaming_on_token_error_swallowed(monkeypatch):
    # A failing on_token callback must not lose the streamed reply.
    sse = ('data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, content=sse.encode()))

    def boom(piece):
        raise RuntimeError("ui gone")

    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}], on_token=boom)
    assert msg["content"] == "hello"


def test_stream_accumulates_tool_calls(monkeypatch):
    sse = (
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", '
        '"type": "function", "function": {"name": "web_search", "arguments": "{\\"q"}}]}}]}\n\n'
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": "\\": \\"x\\"}"}}]}}]}\n\n'
        'data: [DONE]\n\n')
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, content=sse.encode()))
    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}], on_token=lambda t: None)
    assert msg["content"] == ""
    (tc,) = msg["tool_calls"]
    assert tc["id"] == "c9"
    assert tc["function"]["name"] == "web_search"
    assert tc["function"]["arguments"] == '{"q": "x"}'


def test_stream_skips_garbage_lines_and_empty_choices(monkeypatch):
    sse = ('event: ping\n'
           'data: not-json{\n'
           'data: {"choices": []}\n'
           'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, content=sse.encode()))
    msg = providers.call_openai_compatible(
        get_provider("kilo"), "kilo-auto/free",
        [{"role": "user", "content": "hi"}], on_token=lambda t: None)
    assert msg["content"] == "ok"


def test_stream_empty_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, content=b"data: [DONE]\n\n"))
    with pytest.raises(RuntimeError, match="no content"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}], on_token=lambda t: None)


def test_stream_retryable_status_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(503, text="down"))
    with pytest.raises(RuntimeError, match="kilo HTTP 503"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}], on_token=lambda t: None)


def test_stream_400_raises_without_body(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(400, text="nope"))
    with pytest.raises(RuntimeError, match="kilo HTTP 400"):
        providers.call_openai_compatible(
            get_provider("kilo"), "kilo-auto/free",
            [{"role": "user", "content": "hi"}], on_token=lambda t: None)


# ---------------------------------------------------------------------------
# call_cohere — /v2/chat shape
# ---------------------------------------------------------------------------

def _cohere_body(text="cohere answer"):
    return {"message": {"content": [{"type": "text", "text": text},
                                    {"type": "other", "text": "ignored"}]}}


def test_call_cohere_success(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = _json.loads(req.content)
        seen["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, json=_cohere_body())
    _mock_client(monkeypatch, providers, handler)
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    msg = providers.call_cohere(
        get_provider("cohere"), "command-a-03-2025",
        [{"role": "user", "content": "hi"}])
    assert msg["content"] == "cohere answer"
    assert seen["auth"] == "Bearer co-key"
    # Messages are flattened to one user turn.
    assert seen["body"]["messages"] == [{"role": "user", "content": "user: hi"}]


def test_call_cohere_flattens_list_content(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = _json.loads(req.content)
        return httpx.Response(200, json=_cohere_body())
    _mock_client(monkeypatch, providers, handler)
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    providers.call_cohere(
        get_provider("cohere"), "command-a-03-2025",
        [{"role": "user",
          "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert seen["body"]["messages"][0]["content"] == "user: a b"


def test_call_cohere_empty_raises(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json={"message": {"content": []}}))
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    with pytest.raises(RuntimeError, match="empty response"):
        providers.call_cohere(get_provider("cohere"), "command-a-03-2025",
                              [{"role": "user", "content": "hi"}])


def test_call_cohere_retryable_and_400(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(429, text="slow"))
    with pytest.raises(RuntimeError, match="cohere HTTP 429"):
        providers.call_cohere(get_provider("cohere"), "command-a-03-2025",
                              [{"role": "user", "content": "hi"}])
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(401, text="bad key"))
    with pytest.raises(RuntimeError, match="bad key"):
        providers.call_cohere(get_provider("cohere"), "command-a-03-2025",
                              [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# call_provider dispatch + health helpers
# ---------------------------------------------------------------------------

def test_call_provider_dispatches_cohere(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=_cohere_body("via cohere")))
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    msg = providers.call_provider(get_provider("cohere"), "command-a-03-2025",
                                  [{"role": "user", "content": "hi"}])
    assert msg["content"] == "via cohere"


def test_call_provider_progress_callback(monkeypatch):
    _mock_client(monkeypatch, providers,
                 lambda req: httpx.Response(200, json=_ok_message("ok")))
    seen = []
    providers.call_provider(get_provider("kilo"), "kilo-auto/free",
                            [{"role": "user", "content": "hi"}],
                            progress=seen.append)
    assert seen == ["Kilo Gateway: trying kilo-auto/free…"]


def test_health_penalty_unknown_and_thin(monkeypatch):
    assert providers._health_penalty("kilo", "m", {}) == 0
    by_leg = {("kilo", "m"): {"attempts": 2, "success_rate": 0.0}}
    assert providers._health_penalty("kilo", "m", by_leg) == 0


def test_health_penalty_bad_and_stale(monkeypatch):
    by_leg = {("kilo", "m"): {"attempts": 5, "success_rate": 0.2}}
    assert providers._health_penalty("kilo", "m", by_leg) == 1
    by_leg = {("kilo", "m"): {"attempts": 5, "success_rate": 0.9, "stale": True}}
    assert providers._health_penalty("kilo", "m", by_leg) == 2


def test_health_penalty_malformed_row(monkeypatch):
    by_leg = {("kilo", "m"): {"attempts": "many", "success_rate": "bad"}}
    assert providers._health_penalty("kilo", "m", by_leg) == 0


@pytest.mark.parametrize("val,enabled", [
    ("1", True), ("0", False), ("false", False), ("no", False),
    ("off", False), ("yes", True), ("", True),
])
def test_health_order_enabled_env(monkeypatch, val, enabled):
    monkeypatch.setenv("ROUTER_HEALTH_ORDER", val)
    assert providers._health_order_enabled() is enabled


def test_health_order_enabled_default(monkeypatch):
    monkeypatch.delenv("ROUTER_HEALTH_ORDER", raising=False)
    assert providers._health_order_enabled() is True


def test_omni_chain_uncensored_first_off(monkeypatch):
    # With UNCENSORED_FIRST off, the keyless backstop leads the chain.
    import app.guard_policy as gp
    monkeypatch.setattr(gp, "UNCENSORED_FIRST", False)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
    chain = omni_chain()
    names = [p.name for p, _ in chain]
    assert names.index("llm7") < names.index("deepinfra")
    assert names.index("kilo") < names.index("deepinfra")


def test_omni_chain_explicit_unknown_provider_ignored():
    chain = omni_chain("nope:some-model")
    assert chain, "keyless backstop must still be present"
    assert all(p.name != "nope" for p, _ in chain)


def test_providerdef_keyless_property():
    assert get_provider("kilo").keyless is True
    assert get_provider("groq").keyless is False
