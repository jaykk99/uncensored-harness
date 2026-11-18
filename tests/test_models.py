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
