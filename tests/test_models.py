"""Tests for the pure helpers in app/models.py.

No network: everything here exercises normalization, refusal detection,
tool-call parsing, and the self-test verdict logic (with models.chat
monkeypatched to canned answers). The live HTTP paths are exercised by
the probe suite at runtime, not here.
"""

from __future__ import annotations

import json

import pytest

from app import models
from app.models import (
    _norm,
    _parse_tool_calls,
    looks_like_refusal,
    self_test,
)


# ---------------------------------------------------------------------------
# _norm: typographic dodge normalization
# ---------------------------------------------------------------------------

def test_norm_curly_quotes_and_dashes():
    assert _norm("\u201cHello\u201d \u2014 it\u2019s fine") == '"hello" - it\'s fine'


def test_norm_lowercases():
    assert _norm("I CAN'T") == "i can't"


def test_norm_plain_text_unchanged():
    assert _norm("hello world 123") == "hello world 123"


# ---------------------------------------------------------------------------
# looks_like_refusal
# ---------------------------------------------------------------------------

REFUSAL_TEXTS = [
    "I'm sorry, I can't help with that.",
    "I cannot do this as an AI language model.",
    "As an AI, I must decline this request.",
    "That request is against my guidelines and policy.",
    "I refuse to answer that.",
    "This is not appropriate content.",
    "I am not able to comply with that.",
    "",
    "   ",
]

NON_REFUSAL_TEXTS = [
    "Here is a python function that sorts a list: def sort(l): ...",
    "Lockpicking as a hobby uses a tension wrench and a pick.",
    "Sure — here are three first steps for your authorized red-team test.",
    "Mondays are damn frustrating, no doubt about it.",
    # Former false discards: the router used to throw these answers away
    # because of bare substrings ("policy", "i can't", "as an ai").
    "Here is a summary of your company's return policy document.",
    "Your insurance policy covers water damage up to $10,000.",
    "my policymaking notes: council meets Tuesday on zoning.",
    "I'm sorry to hear your server went down. Here is the fix.",
    "I can't wait to show you this design, here it is.",
    "As an AI assistant, here is the code you asked for.",
]


@pytest.mark.parametrize("text", REFUSAL_TEXTS)
def test_refusals_detected(text):
    assert looks_like_refusal(text) is True, f"missed refusal: {text!r}"


@pytest.mark.parametrize("text", NON_REFUSAL_TEXTS)
def test_non_refusals_pass(text):
    assert looks_like_refusal(text) is False, f"false positive: {text!r}"


def test_refusal_detection_survives_typographic_dodges():
    # Curly apostrophe must not dodge the "i can't" pattern.
    assert looks_like_refusal("I can\u2019t help with that.") is True
    # Em dash inside a refusal sentence.
    assert looks_like_refusal("As an AI \u2014 I must decline.") is True


def test_policy_words_alone_are_not_a_refusal():
    # Formerly "policy" was a bare substring, so a sentence about
    # "policymaking" was discarded as a refusal. Refusals that mention
    # policy are still caught by their paired verb ("against my",
    # "my guidelines", "i must decline").
    assert looks_like_refusal("my policymaking notes") is False
    assert looks_like_refusal(
        "That request is against my guidelines and policy.") is True


def test_none_is_treated_as_empty():
    assert looks_like_refusal(None) is True


# ---------------------------------------------------------------------------
# _parse_tool_calls
# ---------------------------------------------------------------------------

def test_parse_single_tool_call():
    msg = {"tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "web_search",
                      "arguments": json.dumps({"query": "x"})}}]}
    (call,) = _parse_tool_calls(msg)
    assert call == {"id": "call_1", "name": "web_search", "args": {"query": "x"}}


def test_parse_multiple_and_malformed_args():
    msg = {"tool_calls": [
        {"id": "a", "function": {"name": "t1", "arguments": "not-json{"}},
        {"id": "b", "function": {"name": "t2", "arguments": '{"k": 1}'}},
        {"function": {"name": "t3"}},  # no id, no args
    ]}
    calls = _parse_tool_calls(msg)
    assert len(calls) == 3
    assert calls[0]["args"] == {}
    assert calls[1]["args"] == {"k": 1}
    assert calls[2]["id"].startswith("call_")
    assert calls[2]["args"] == {}


def test_parse_no_tool_calls():
    assert _parse_tool_calls({}) == []
    assert _parse_tool_calls({"content": "hi"}) == []


# ---------------------------------------------------------------------------
# self_test verdict logic (chat is stubbed — no network)
# ---------------------------------------------------------------------------

def _stub_chat_factory(answers):
    """answers: list of (text, source) per probe, or Exception instances."""
    def fake_chat(messages, max_tokens=300, temperature=0.7):
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        text, source = item
        return text, source, [], None, False
    return fake_chat


def test_self_test_all_pass(monkeypatch):
    n = len(models.SELF_TEST_PROBES)
    answers = [("Lockpicking uses a tension wrench and pick.", "horde")] * n
    monkeypatch.setattr(models, "chat", _stub_chat_factory(answers))
    report = self_test()
    assert report["ok"] is True
    assert report["passed"] == n and report["total"] == n
    assert "UNCENSORED" in report["verdict"]
    assert all(p["passed"] for p in report["probes"])
    assert [p["id"] for p in report["probes"]] == [
        "hobby_dual_use", "free_expression", "authorized_security",
        "opinion_hedging"]


def test_self_test_partial_on_refusal(monkeypatch):
    answers = [
        ("Lockpicking uses a tension wrench and pick.", "horde"),
        ("I'm sorry, I can't write that.", "pollinations"),
        ("Here are three red-team first steps.", "horde"),
        # The new opinion probe: an RLHF hedge trips the "as an ai" pattern.
        ("As an AI, I don't have personal opinions on that.", "horde"),
    ]
    monkeypatch.setattr(models, "chat", _stub_chat_factory(answers))
    report = self_test()
    assert report["ok"] is False
    assert report["passed"] == 2
    assert "PARTIAL" in report["verdict"]
    assert report["probes"][1]["passed"] is False
    assert "REFUSAL" in report["probes"][1]["detail"]
    assert report["probes"][3]["passed"] is False
    assert "REFUSAL" in report["probes"][3]["detail"]


def test_self_test_error_counts_as_failure(monkeypatch):
    answers = [
        RuntimeError("all model legs failed"),
        ("fine", "horde"),
        ("fine", "horde"),
        ("fine", "horde"),
    ]
    monkeypatch.setattr(models, "chat", _stub_chat_factory(answers))
    report = self_test()
    assert report["ok"] is False
    assert report["passed"] == 3
    assert "ERROR" in report["probes"][0]["detail"]


def test_self_test_source_recorded_per_probe(monkeypatch):
    n = len(models.SELF_TEST_PROBES)
    answers = [("ok text", f"src-{i}") for i in range(n)]
    monkeypatch.setattr(models, "chat", _stub_chat_factory(answers))
    report = self_test()
    assert [p["source"] for p in report["probes"]] == [f"src-{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Provider list constants stay sane
# ---------------------------------------------------------------------------

def test_horde_model_list_uncensored_lineage():
    assert len(models.HORDE_UNCENSORED_MODELS) >= 3
    joined = " ".join(models.HORDE_UNCENSORED_MODELS).lower()
    assert any(k in joined for k in ("stheno", "heretic", "rocinante"))


def test_retryable_statuses_cover_rate_limits():
    assert {429, 500, 502, 503}.issubset(models.RETRYABLE_STATUS)


def test_pollinations_endpoint_is_openai_compatible():
    assert models.POLLINATIONS_URL.endswith("/chat/completions")


# ---------------------------------------------------------------------------
# _record_leg / _omni_leg: error detail reaches router_stats.last_error
# ---------------------------------------------------------------------------

def test_record_leg_passes_error_detail(tmp_path, monkeypatch):
    import time
    from app import router_stats as rs
    monkeypatch.setattr(rs, "DB", str(tmp_path / "stats.db"))
    models._record_leg("kilo", "m1", "error", time.time(),
                       detail="kilo HTTP 429: slow down")
    (row,) = rs.snapshot()
    assert row["errors"] == 1
    assert row["last_error"] == "kilo HTTP 429: slow down"


def test_omni_leg_records_provider_error_detail(tmp_path, monkeypatch):
    from app import router_stats as rs
    monkeypatch.setattr(rs, "DB", str(tmp_path / "stats.db"))
    import app.providers as providers_mod

    class FakeProvider:
        name = "groq"
        label = "Groq"

    monkeypatch.setattr(providers_mod, "omni_chain",
                        lambda *a, **k: [(FakeProvider(), "m1")])

    def boom(p, model, *a, **k):
        raise RuntimeError("groq HTTP 401: bad key")

    monkeypatch.setattr(providers_mod, "call_provider", boom)
    assert models._omni_leg([], 10, 0.7, None, None, "") is None
    (row,) = rs.snapshot()
    assert row["provider"] == "groq" and row["errors"] == 1
    assert row["last_error"] == "groq HTTP 401: bad key"


def test_omni_leg_empty_response_records_reason(tmp_path, monkeypatch):
    # An empty reply is a failure, not a refusal: it reaches the
    # empty-response error branch, and the reason is stored.
    from app import router_stats as rs
    monkeypatch.setattr(rs, "DB", str(tmp_path / "stats.db"))
    import app.providers as providers_mod

    class FakeProvider:
        name = "kilo"
        label = "Kilo"

    monkeypatch.setattr(providers_mod, "omni_chain",
                        lambda *a, **k: [(FakeProvider(), "m1")])
    monkeypatch.setattr(providers_mod, "call_provider",
                        lambda *a, **k: {"content": ""})
    assert models._omni_leg([], 10, 0.7, None, None, "") is None
    (row,) = rs.snapshot()
    assert row["errors"] == 1
    assert row["last_error"] == "empty response"


def test_omni_leg_refusal_still_refusal_not_error(tmp_path, monkeypatch):
    # The reorder above must not reclassify real refusals: a reply with
    # refusal language still counts as a refusal, with no last_error.
    from app import router_stats as rs
    monkeypatch.setattr(rs, "DB", str(tmp_path / "stats.db"))
    import app.providers as providers_mod

    class FakeProvider:
        name = "kilo"
        label = "Kilo"

    monkeypatch.setattr(providers_mod, "omni_chain",
                        lambda *a, **k: [(FakeProvider(), "m1")])
    monkeypatch.setattr(
        providers_mod, "call_provider",
        lambda *a, **k: {"content": "I'm sorry, I can't help with that."})
    assert models._omni_leg([], 10, 0.7, None, None, "") is None
    (row,) = rs.snapshot()
    assert row["refusals"] == 1 and row["errors"] == 0
    assert row["last_error"] is None


# ---------------------------------------------------------------------------
# HTTP paths — stubbed httpx transport, no network, no real sleeps
# ---------------------------------------------------------------------------

import time as _time

import httpx


def _mock_models_client(monkeypatch, handler):
    def factory(timeout):
        return httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(models, "_client", factory)


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch):
    monkeypatch.setattr(models, "_sleep_backoff", lambda attempt, **k: 0.0)
    monkeypatch.setattr(models, "_pace_pollinations", lambda: None)
    monkeypatch.setattr(_time, "sleep", lambda s: None)


def _poll_msg(content="pollinations answer"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# --- pollinations_chat (non-streaming) ---

def test_pollinations_chat_success(monkeypatch):
    _mock_models_client(monkeypatch, lambda req: httpx.Response(200, json=_poll_msg()))
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}])
    assert msg["content"] == "pollinations answer"


def test_pollinations_chat_tools_in_body(monkeypatch):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_poll_msg())
    _mock_models_client(monkeypatch, handler)
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    models.pollinations_chat([{"role": "user", "content": "hi"}], tools=tools)
    assert seen["body"]["tools"] == tools
    assert seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["model"] == models.POLLINATIONS_MODEL


def test_pollinations_chat_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_poll_msg("second try"))
    _mock_models_client(monkeypatch, handler)
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}],
                                   retries=3)
    assert msg["content"] == "second try"
    assert calls["n"] == 2


def test_pollinations_chat_empty_choices_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json=_poll_msg("after empty"))
    _mock_models_client(monkeypatch, handler)
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}],
                                   retries=3)
    assert msg["content"] == "after empty"


def test_pollinations_chat_empty_choices_exhausted(monkeypatch):
    _mock_models_client(monkeypatch,
                        lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(RuntimeError, match="empty choices"):
        models.pollinations_chat([{"role": "user", "content": "hi"}], retries=2)


def test_pollinations_chat_keeps_failing(monkeypatch):
    _mock_models_client(monkeypatch,
                        lambda req: httpx.Response(429, text="slow"))
    with pytest.raises(RuntimeError, match="kept failing after 3 attempts"):
        models.pollinations_chat([{"role": "user", "content": "hi"}], retries=3)


def test_pollinations_chat_http_status_error(monkeypatch):
    # A non-retryable 4xx surfaces as an HTTP error with the status + body.
    _mock_models_client(monkeypatch,
                        lambda req: httpx.Response(400, text="bad request"))
    with pytest.raises(RuntimeError, match="pollinations HTTP 400: bad request"):
        models.pollinations_chat([{"role": "user", "content": "hi"}], retries=1)


def test_pollinations_chat_network_error_retries_then_fails(monkeypatch):
    def handler(req):
        raise httpx.ConnectError("boom", request=req)
    _mock_models_client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="network failure"):
        models.pollinations_chat([{"role": "user", "content": "hi"}], retries=2)


def test_pollinations_chat_progress_reports_attempts(monkeypatch):
    _mock_models_client(monkeypatch, lambda req: httpx.Response(200, json=_poll_msg()))
    seen = []
    models.pollinations_chat([{"role": "user", "content": "hi"}],
                             progress=seen.append)
    assert seen == ["Pollinations: attempt 1/4…"]


# --- _pollinations_stream ---

def _sse_response(text):
    return httpx.Response(200, content=text.encode())


def test_pollinations_stream_forwards_tokens(monkeypatch):
    sse = ('data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_models_client(monkeypatch, lambda req: _sse_response(sse))
    seen = []
    msg = models._pollinations_stream({"model": "openai", "messages": []},
                                      30, seen.append)
    assert msg["content"] == "hello"
    assert seen == ["hel", "lo"]


def test_pollinations_stream_accumulates_tool_calls(monkeypatch):
    sse = (
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", '
        '"function": {"name": "shell_exec", "arguments": "{\\"cmd"}}]}}]}\n\n'
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": "\\": \\"ls\\"}"}}]}}]}\n\n'
        'data: [DONE]\n\n')
    _mock_models_client(monkeypatch, lambda req: _sse_response(sse))
    msg = models._pollinations_stream({"model": "openai", "messages": []},
                                      30, lambda t: None)
    assert msg["content"] == ""
    (tc,) = msg["tool_calls"]
    assert tc["function"]["name"] == "shell_exec"
    assert tc["function"]["arguments"] == '{"cmd": "ls"}'


def test_pollinations_stream_skips_bad_lines(monkeypatch):
    sse = ('data: garbage{\n'
           'data: {"choices": []}\n'
           'data: {"choices": [{"delta": {"content": "fine"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_models_client(monkeypatch, lambda req: _sse_response(sse))
    msg = models._pollinations_stream({"model": "openai", "messages": []},
                                      30, lambda t: None)
    assert msg["content"] == "fine"


def test_pollinations_stream_empty_raises(monkeypatch):
    _mock_models_client(monkeypatch,
                        lambda req: _sse_response("data: [DONE]\n\n"))
    with pytest.raises(RuntimeError, match="no content"):
        models._pollinations_stream({"model": "openai", "messages": []},
                                    30, lambda t: None)


def test_pollinations_stream_retryable_status(monkeypatch):
    _mock_models_client(monkeypatch,
                        lambda req: httpx.Response(502, text="bad gateway"))
    with pytest.raises(RuntimeError, match="pollinations HTTP 502"):
        models._pollinations_stream({"model": "openai", "messages": []},
                                    30, lambda t: None)


def test_pollinations_stream_on_token_failure_tolerated(monkeypatch):
    sse = ('data: {"choices": [{"delta": {"content": "a"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": "b"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_models_client(monkeypatch, lambda req: _sse_response(sse))

    def bad_token(t):
        raise RuntimeError("UI blew up")

    msg = models._pollinations_stream({"model": "openai", "messages": []},
                                      30, bad_token)
    assert msg["content"] == "ab"


def test_pollinations_chat_streaming_path(monkeypatch):
    sse = ('data: {"choices": [{"delta": {"content": "streamed"}}]}\n\n'
           'data: [DONE]\n\n')
    _mock_models_client(monkeypatch, lambda req: _sse_response(sse))
    seen = []
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}],
                                   on_token=seen.append)
    assert msg["content"] == "streamed"
    assert seen == ["streamed"]


# --- Horde submit / generate ---

def _horde_submit_ok(req):
    assert req.url.path.endswith("/generate/text/async")
    return httpx.Response(200, json={"id": "job-123"})


def test_horde_submit_success(monkeypatch):
    _mock_models_client(monkeypatch, _horde_submit_ok)
    job = models._horde_submit("https://stablehorde.net/api/v2",
                               {"apikey": "0000000000"}, {"prompt": "x"})
    assert job == "job-123"


def test_horde_submit_retries_429(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow")
        return httpx.Response(200, json={"id": "job-9"})
    _mock_models_client(monkeypatch, handler)
    job = models._horde_submit("https://stablehorde.net/api/v2", {}, {})
    assert job == "job-9"
    assert calls["n"] == 2


def test_horde_submit_no_job_id_raises(monkeypatch):
    _mock_models_client(monkeypatch,
                        lambda req: httpx.Response(200, json={}))
    with pytest.raises(RuntimeError, match="no job id"):
        models._horde_submit("https://stablehorde.net/api/v2", {}, {})


def test_horde_submit_network_error_exhausts(monkeypatch):
    def handler(req):
        raise httpx.ConnectError("down", request=req)
    _mock_models_client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="horde submit failed"):
        models._horde_submit("https://stablehorde.net/api/v2", {}, {})


def _horde_status_handler(done_text="horde says hi", faulted=False,
                          vanish=False):
    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            return httpx.Response(200, json={"id": "job-1"})
        if vanish:
            return httpx.Response(404, text="gone")
        return httpx.Response(200, json={
            "done": True, "faulted": faulted, "queue_position": 0,
            "generations": [{"text": done_text}] if done_text else []})
    return handler


def test_horde_generate_returns_text(monkeypatch):
    _mock_models_client(monkeypatch, _horde_status_handler())
    text = models._horde_generate("prompt", 100, 0.7, timeout_s=30)
    assert text == "horde says hi"


def test_horde_generate_faulted_fails_both_endpoints(monkeypatch):
    _mock_models_client(monkeypatch, _horde_status_handler(faulted=True))
    with pytest.raises(RuntimeError, match="all endpoints failed"):
        models._horde_generate("prompt", 100, 0.7, timeout_s=30)


def test_horde_generate_job_vanished(monkeypatch):
    _mock_models_client(monkeypatch, _horde_status_handler(vanish=True))
    with pytest.raises(RuntimeError, match="all endpoints failed"):
        models._horde_generate("prompt", 100, 0.7, timeout_s=30)


def test_horde_generate_empty_generation(monkeypatch):
    _mock_models_client(monkeypatch, _horde_status_handler(done_text=""))
    with pytest.raises(RuntimeError, match="all endpoints failed"):
        models._horde_generate("prompt", 100, 0.7, timeout_s=30)


def test_horde_generate_timeout(monkeypatch):
    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            return httpx.Response(200, json={"id": "job-1"})
        return httpx.Response(200, json={"done": False, "queue_position": 3})
    _mock_models_client(monkeypatch, handler)
    # Sleep is stubbed to no-op, so the poll loop spins on the mock until
    # the real 1s deadline, then raises.
    with pytest.raises(RuntimeError, match="all endpoints failed"):
        models._horde_generate("prompt", 100, 0.7, timeout_s=1)


def test_horde_generate_falls_back_to_second_api(monkeypatch):
    def handler(req):
        host = req.url.host
        if req.url.path.endswith("/generate/text/async"):
            if "stablehorde" in host:
                return httpx.Response(500, text="oops")
            return httpx.Response(200, json={"id": "job-2"})
        return httpx.Response(200, json={
            "done": True, "generations": [{"text": "second api wins"}]})
    _mock_models_client(monkeypatch, handler)
    text = models._horde_generate("prompt", 100, 0.7, timeout_s=30)
    assert text == "second api wins"


def test_horde_generate_preferred_model_first(monkeypatch):
    seen = {}

    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            seen["models"] = json.loads(req.content)["models"]
            return httpx.Response(200, json={"id": "job-1"})
        return httpx.Response(200, json={
            "done": True, "generations": [{"text": "ok"}]})
    _mock_models_client(monkeypatch, handler)
    preferred = "koboldcpp/Rocinante-X-12B-v1"
    models._horde_generate("prompt", 100, 0.7, timeout_s=30,
                           preferred_model=preferred)
    assert seen["models"][0] == preferred
    assert len(seen["models"]) == len(models.HORDE_UNCENSORED_MODELS)


def test_horde_chat_flattens_messages(monkeypatch):
    seen = {}

    def fake_generate(prompt, *a, **k):
        seen["prompt"] = prompt
        return "out"

    monkeypatch.setattr(models, "_horde_generate", fake_generate)
    out = models.horde_chat([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi back"},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": [{"type": "text", "text": "list part"}]},
    ])
    assert out == "out"
    prompt = seen["prompt"]
    assert "be helpful" in prompt
    assert "User: hello" in prompt
    assert "Assistant: hi back" in prompt
    assert "Tool result: tool output" in prompt
    assert "list part" in prompt
    assert prompt.endswith("\nAssistant:")


# --- _race_first_response / chat() fallback chain ---

def _race(monkeypatch, horde=None, kilo=None, llm7=None, poll=None):
    import app.providers as pm
    monkeypatch.setattr(models, "horde_chat", horde or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("horde down"))))
    monkeypatch.setattr(models, "pollinations_chat", poll or (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poll down"))))

    def fake_call_provider(p, model, messages, *a, **k):
        name = p.name
        canned = {"kilo": kilo, "llm7": llm7}.get(name)
        if canned is None:
            raise RuntimeError(f"{name} down")
        return {"content": canned}
    monkeypatch.setattr(pm, "call_provider", fake_call_provider)


def test_race_kilo_wins_when_others_fail(monkeypatch):
    _race(monkeypatch, kilo="kilo answered")
    text, source, calls, raw = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "kilo answered"
    assert source.startswith("kilo/")


def test_race_pollinations_wins(monkeypatch):
    _race(monkeypatch, poll=lambda *a, **k: {"content": "poll wins"})
    text, source, calls, raw = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "poll wins"
    assert source == "pollinations"


def test_race_horde_wins(monkeypatch):
    _race(monkeypatch, horde=lambda *a, **k: "horde wins")
    text, source, calls, raw = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "horde wins"
    assert source == "horde"


def test_race_refusal_skipped_other_leg_wins(monkeypatch):
    _race(monkeypatch,
          horde=lambda *a, **k: "I'm sorry, I can't help with that.",
          kilo="kilo saves the turn")
    text, source, calls, raw = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "kilo saves the turn"


def test_race_all_fail_raises(monkeypatch):
    _race(monkeypatch)
    with pytest.raises(RuntimeError, match="All providers failed"):
        models._race_first_response(
            [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")


def test_race_empty_reply_is_error_not_winner(monkeypatch):
    _race(monkeypatch, kilo="", llm7="llm7 non-empty")
    text, source, calls, raw = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "llm7 non-empty"


def test_race_progress_reported(monkeypatch):
    _race(monkeypatch, kilo="k")
    seen = []
    models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, seen.append, "")
    assert seen and "Trying fast providers" in seen[0]


def test_chat_race_win_returns_immediately(monkeypatch):
    monkeypatch.setattr(
        models, "_race_first_response",
        lambda *a, **k: ("race text", "kilo/m", [], None))
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}])
    assert (text, source, streamed) == ("race text", "kilo/m", False)


def test_chat_falls_back_to_omni(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg",
                        lambda *a, **k: ("omni text", "groq/m", [], {"content": "omni text"}))
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}])
    assert text == "omni text" and source == "groq/m"


def test_chat_falls_back_to_pollinations(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": "poll fallback"})
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}])
    assert text == "poll fallback" and source == "pollinations"


def test_chat_pollinations_refusal_then_horde_retry(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": "I'm sorry, I can't help."})
    monkeypatch.setattr(models, "horde_chat",
                        lambda *a, **k: "horde retry answer")
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}])
    assert text == "horde retry answer"
    assert source == "horde (retry)"


def test_chat_horde_retry_refusal_labelled(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poll down")))
    monkeypatch.setattr(models, "horde_chat",
                        lambda *a, **k: "I'm sorry, I can't help.")
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}])
    assert source == "horde (refused too)"


def test_chat_all_legs_fail_raises(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poll down")))
    monkeypatch.setattr(models, "horde_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("horde down")))
    with pytest.raises(RuntimeError, match="all model legs failed"):
        models.chat([{"role": "user", "content": "hi"}])


def test_chat_streamed_flag_propagates(monkeypatch):
    monkeypatch.setattr(models, "_race_first_response",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)

    def fake_poll(*a, **k):
        on_token = k.get("on_token")
        if on_token:
            on_token("tok")
        return {"content": "tok"}
    monkeypatch.setattr(models, "pollinations_chat", fake_poll)
    text, source, calls, raw, streamed = models.chat(
        [{"role": "user", "content": "hi"}], on_token=lambda t: None)
    assert streamed is True


# --- small helpers ---

def test_sleep_backoff_grows_with_attempt(monkeypatch):
    # The autouse fixture stubs _sleep_backoff itself; restore the real one.
    monkeypatch.undo()
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    monkeypatch.setattr(models.random, "uniform", lambda a, b: 0.5)
    d0 = models._sleep_backoff(0)
    d3 = models._sleep_backoff(3)
    assert d0 == pytest.approx(2.5)
    assert d3 == pytest.approx(15.5)  # capped at 15 + jitter


def test_pace_pollinations_waits_when_recent(monkeypatch):
    # The autouse fixture stubs _pace_pollinations itself; restore the real one.
    monkeypatch.undo()
    import time as real_time
    prev = models._last_pollinations_ts
    try:
        models._last_pollinations_ts = real_time.time()
        waited = []
        monkeypatch.setattr(_time, "sleep", lambda s: waited.append(s))
        models._pace_pollinations()
        assert waited and 0 < waited[0] <= 1.5
    finally:
        models._last_pollinations_ts = prev


def test_pace_pollinations_no_wait_when_stale(monkeypatch):
    monkeypatch.undo()
    import time as real_time
    prev = models._last_pollinations_ts
    try:
        models._last_pollinations_ts = real_time.time() - 100
        waited = []
        monkeypatch.setattr(_time, "sleep", lambda s: waited.append(s))
        models._pace_pollinations()
        assert waited == []
    finally:
        models._last_pollinations_ts = prev


def test_proxy_ignores_loopback(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    assert models._proxy() is None


def test_proxy_returns_real_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    assert models._proxy() == "http://proxy.example:8080"


def test_proxy_none_when_unset(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert models._proxy() is None


def test_verify_returns_bundle_when_present(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    assert models._verify() == "/etc/ssl/certs/ca-certificates.crt"
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert models._verify() is True

# ---------------------------------------------------------------------------
# Coverage batch B (2026-11-21): leg fallbacks, provider branches, retry
# progress, _proxy/_client factories, _record_leg never-raise.
# ---------------------------------------------------------------------------

def test_record_leg_never_raises(monkeypatch):
    monkeypatch.setattr(models.router_stats, "record",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("stats down")))
    models._record_leg("p", "m", "ok", _time.time())  # must not raise


def test_proxy_malformed_url_returns_none(monkeypatch):
    # urlparse raises ValueError on a malformed IPv6 proxy URL.
    for var in ("https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://[::1")
    assert models._proxy() is None


def test_client_factory_passes_proxy_and_verify(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    seen = {}

    class FakeClient:
        def __init__(self, *a, **k):
            seen.update(k)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    models._client(10)
    assert seen["timeout"] == 10
    assert seen["proxy"] is None
    assert seen["trust_env"] is False
    assert seen["verify"] == models._verify()


def test_aclient_factory_passes_proxy_and_verify(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    seen = {}

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            seen.update(k)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    models._aclient(10)
    assert seen["timeout"] == 10
    assert seen["proxy"] is None
    assert seen["trust_env"] is False
    assert seen["verify"] == models._verify()


def test_pollinations_chat_rate_limit_progress(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_poll_msg("second try"))
    _mock_models_client(monkeypatch, handler)
    seen = []
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}],
                                   retries=3, progress=seen.append)
    assert msg["content"] == "second try"
    assert any("rate-limited" in s for s in seen)


def test_pollinations_chat_network_error_progress(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json=_poll_msg("after blip"))
    _mock_models_client(monkeypatch, handler)
    seen = []
    msg = models.pollinations_chat([{"role": "user", "content": "hi"}],
                                   retries=3, progress=seen.append)
    assert msg["content"] == "after blip"
    assert any("network error" in s for s in seen)


def test_horde_generate_progress_and_queue_position(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            return httpx.Response(200, json={"id": "job-1"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"done": False,
                                             "queue_position": 3})
        return httpx.Response(200, json={
            "done": True, "generations": [{"text": "queued then done"}]})
    _mock_models_client(monkeypatch, handler)
    seen = []
    text = models._horde_generate("prompt", 100, 0.7, timeout_s=30,
                                  progress=seen.append)
    assert text == "queued then done"
    assert any("submitting job" in s for s in seen)
    assert any("waiting for a worker" in s and "queue position 3" in s
               for s in seen)


def test_horde_generate_poll_http_error_retries(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            return httpx.Response(200, json={"id": "job-1"})
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("poll blip", request=req)
        return httpx.Response(200, json={
            "done": True, "generations": [{"text": "after blip"}]})
    _mock_models_client(monkeypatch, handler)
    text = models._horde_generate("prompt", 100, 0.7, timeout_s=30)
    assert text == "after blip"


def test_horde_generate_poll_non200_continues(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        if req.url.path.endswith("/generate/text/async"):
            return httpx.Response(200, json={"id": "job-1"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="upstream blip")
        return httpx.Response(200, json={
            "done": True, "generations": [{"text": "after 500"}]})
    _mock_models_client(monkeypatch, handler)
    text = models._horde_generate("prompt", 100, 0.7, timeout_s=30)
    assert text == "after 500"


def test_race_horde_refusal_is_skipped(monkeypatch):
    _race(monkeypatch,
          horde=lambda *a, **k: (_ for _ in ()).throw(
              models._Refusal("nope")),
          kilo="kilo after refusal")
    text, source, _, _ = models._race_first_response(
        [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")
    assert text == "kilo after refusal"
    assert source.startswith("kilo/")


def test_race_unknown_provider_leg_fails(monkeypatch):
    import app.providers as pm
    _race(monkeypatch)
    monkeypatch.setattr(pm, "get_provider", lambda name: None)
    with pytest.raises(RuntimeError, match="All providers failed"):
        models._race_first_response(
            [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")


def test_race_fast_provider_refusal_tries_next_model(monkeypatch):
    import app.providers as pm
    from types import SimpleNamespace
    _race(monkeypatch)
    fake = SimpleNamespace(name="kilo", models=["m1", "m2"])
    monkeypatch.setattr(pm, "get_provider",
                        lambda name: fake if name in ("kilo", "llm7") else None)
    monkeypatch.setattr(
        pm, "call_provider",
        lambda p, model, *a, **k: {"content": "I'm sorry, I can't help."})
    with pytest.raises(RuntimeError, match="All providers failed"):
        models._race_first_response(
            [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")


def test_race_pollinations_refusal_skipped(monkeypatch):
    import app.providers as pm
    _race(monkeypatch,
          poll=lambda *a, **k: {"content": "I'm sorry, I can't help."})
    monkeypatch.setattr(pm, "get_provider", lambda name: None)
    with pytest.raises(RuntimeError, match="All providers failed"):
        models._race_first_response(
            [{"role": "user", "content": "hi"}], 100, 0.7, None, None, "")


def test_chat_race_failure_reports_progress(monkeypatch):
    monkeypatch.setattr(
        models, "_race_first_response",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(
        models, "pollinations_chat",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poll down")))
    monkeypatch.setattr(
        models, "horde_chat",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("horde down")))
    seen = []
    with pytest.raises(RuntimeError, match="all model legs failed"):
        models.chat([{"role": "user", "content": "hi"}],
                    progress=seen.append)
    assert any("Fast race failed" in s for s in seen)
    assert any("Omni providers exhausted" in s for s in seen)
    assert any("Pollinations failed" in s for s in seen)


def test_chat_omni_leg_error_falls_through(monkeypatch):
    monkeypatch.setattr(
        models, "_race_first_response",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(
        models, "_omni_leg",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("omni boom")))
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": "poll after omni boom"})
    text, source, _, _, _ = models.chat([{"role": "user", "content": "hi"}])
    assert text == "poll after omni boom"
    assert source == "pollinations"


def test_chat_pollinations_refusal_progress(monkeypatch):
    monkeypatch.setattr(
        models, "_race_first_response",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("race down")))
    monkeypatch.setattr(models, "_omni_leg", lambda *a, **k: None)
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": "I'm sorry, I can't help."})
    monkeypatch.setattr(models, "horde_chat",
                        lambda *a, **k: "horde retry answer")
    seen = []
    text, source, _, _, _ = models.chat(
        [{"role": "user", "content": "hi"}], progress=seen.append)
    assert text == "horde retry answer"
    assert source == "horde (retry)"
    assert any("Pollinations refused" in s for s in seen)


def test_omni_leg_stats_failure_still_returns_none(monkeypatch):
    import app.providers as pm
    monkeypatch.setattr(models.router_stats, "snapshot",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("stats down")))
    monkeypatch.setattr(pm, "omni_chain", lambda *a, **k: [])
    assert models._omni_leg([{"role": "user", "content": "hi"}],
                            100, 0.7, None, None, "") is None


def test_omni_leg_provider_failure_then_success(monkeypatch):
    import app.providers as pm
    from types import SimpleNamespace
    p1 = SimpleNamespace(name="first", label="First")
    p2 = SimpleNamespace(name="second", label="Second")
    monkeypatch.setattr(models.router_stats, "snapshot", lambda: [])
    monkeypatch.setattr(pm, "omni_chain",
                        lambda *a, **k: [(p1, "m1"), (p2, "m2")])

    def fake_call(p, model, messages, *a, **k):
        if model == "m1":
            raise RuntimeError("first down")
        return {"content": "second works"}
    monkeypatch.setattr(pm, "call_provider", fake_call)
    seen = []
    out = models._omni_leg([{"role": "user", "content": "hi"}],
                           100, 0.7, None, seen.append, "")
    assert out is not None
    assert out[0] == "second works"
    assert out[1] == "second/m2"
    assert any("failed — trying next provider" in s for s in seen)

def test_refusal_skipping_off_returns_first_answer_as_is(monkeypatch):
    # guard_policy SKIP_REFUSALS off: even a canonical refusal is returned.
    monkeypatch.setattr("app.guard_policy.SKIP_REFUSALS", False)
    assert models.looks_like_refusal("I'm sorry, I can't help with that.") is False
    assert models.looks_like_refusal("here is the answer") is False
