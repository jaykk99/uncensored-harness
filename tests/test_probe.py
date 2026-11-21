"""Tests for app/probe.py — the per-provider uncensored probe suite.

The network legs (models.horde_chat, models.pollinations_chat,
providers.call_provider) are stubbed; probe logic (per-model verdicts,
skip-not-live, refusal detection, CLI) is what's under test.
"""

from __future__ import annotations

import json
import sys

import pytest

from app import models, probe as probe_mod
from app import providers as providers_mod


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for p in providers_mod.ALL_PROVIDERS:
        if p.key_env:
            monkeypatch.delenv(p.key_env, raising=False)


@pytest.fixture(autouse=True)
def tmp_history_db(tmp_path, monkeypatch):
    # run_suite records every run to probe history; point it at a tmp
    # db so the suite tests never write into the repo's real harness.db.
    from app import probe_history as history_mod
    monkeypatch.setattr(history_mod, "DB", str(tmp_path / "probe.db"))


def _ok_text(prompt):
    return "Here is a straight answer with no hedging whatsoever."


def _refuse_text(prompt):
    return "I'm sorry, I can't help with that."


@pytest.fixture
def stub_all_ok(monkeypatch):
    monkeypatch.setattr(models, "horde_chat",
                        lambda *a, **k: _ok_text(a[0] if a else ""))
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": _ok_text("")})
    def fake_call(p, model, messages, **k):
        return {"content": _ok_text("")}
    monkeypatch.setattr(providers_mod, "call_provider", fake_call)


def test_probe_one_model_all_pass(stub_all_ok):
    entry = probe_mod.probe_one_model("llm7", "llm7:codestral-latest")
    assert entry["passed"] == entry["total"] == len(models.SELF_TEST_PROBES)
    assert entry["error"] == ""
    assert all(p["passed"] for p in entry["probes"])
    assert entry["elapsed_s"] >= 0


def test_probe_one_model_refusal_detected(monkeypatch):
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": _refuse_text("")})
    entry = probe_mod.probe_one_model("pollinations", "pollinations:openai")
    assert entry["passed"] == 0
    assert "REFUSAL" in entry["probes"][0]["detail"]


def test_probe_one_model_error_recorded_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider is down")
    monkeypatch.setattr(models, "horde_chat", boom)
    entry = probe_mod.probe_one_model("horde", "horde:koboldcpp/L3-8B-Stheno-v3.2")
    assert entry["passed"] == 0
    assert "provider is down" in entry["error"]


def test_probe_via_provider_uses_bare_model_id(monkeypatch):
    seen = {}

    def fake_call(p, model, messages, **k):
        seen["provider"] = p.name
        seen["model"] = model
        return {"content": _ok_text("")}

    monkeypatch.setattr(providers_mod, "call_provider", fake_call)
    entry = probe_mod.probe_one_model("kilo", "kilo:kilo-auto/free")
    assert seen == {"provider": "kilo", "model": "kilo-auto/free"}
    assert entry["passed"] == len(models.SELF_TEST_PROBES)


def test_run_suite_skips_unkeyed_providers(stub_all_ok):
    report = probe_mod.run_suite()
    assert report["probed"] > 0
    # Groq has no key in this env -> skipped, not failed.
    skipped_ids = [s["id"] for s in report["skipped"]]
    assert any(i.startswith("groq:") for i in skipped_ids)
    assert report["ok"] is True
    assert "ALL CLEAR" in report["verdict"]


def test_run_suite_provider_filter(stub_all_ok):
    report = probe_mod.run_suite(provider="llm7")
    assert report["probed"] == len(providers_mod.LLM7_MODELS)
    assert all(r["provider"] == "llm7" for r in report["results"])


def test_run_suite_single_model(stub_all_ok):
    report = probe_mod.run_suite(model_id="pollinations:openai")
    assert report["probed"] == 1
    assert report["results"][0]["model"] == "pollinations:openai"


def test_run_suite_partial_verdict(monkeypatch, stub_all_ok):
    # Make one model refuse while the rest pass.
    orig = providers_mod.call_provider

    def fake_call(p, model, messages, **k):
        if model == "mistral-Nemo-Instruct-2407":
            return {"content": _refuse_text("")}
        return {"content": _ok_text("")}

    monkeypatch.setattr(providers_mod, "call_provider", fake_call)
    report = probe_mod.run_suite(provider="llm7")
    assert report["ok"] is False
    assert report["passed_models"] == len(providers_mod.LLM7_MODELS) - 1
    assert "PARTIAL" not in report["verdict"]  # suite verdict is a fraction
    assert "/" in report["verdict"]


def test_run_suite_records_probe_history(stub_all_ok):
    from app import probe_history as history_mod
    report = probe_mod.run_suite(provider="llm7")
    assert report["probed"] == len(providers_mod.LLM7_MODELS)
    latest = history_mod.latest_per_model()
    assert len(latest) == len(providers_mod.LLM7_MODELS)
    assert all(r["clean"] for r in latest)
    runs = history_mod.recent_runs()
    assert len(runs) == 1
    assert runs[0]["provider"] == "llm7"
    assert runs[0]["probed"] == len(providers_mod.LLM7_MODELS)


def test_summarize_marks(capsys, stub_all_ok):
    report = probe_mod.run_suite(provider="llm7")
    text = probe_mod.summarize(report)
    assert "[PASS]" in text and "[SKIP]" not in text  # llm7 all live
    assert "probe suite:" in text


def test_cli_json_output(monkeypatch, capsys, stub_all_ok):
    rc = probe_mod.main(["--provider", "llm7", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["ok"] is True and data["probed"] == len(providers_mod.LLM7_MODELS)


def test_cli_exit_code_on_failure(monkeypatch):
    monkeypatch.setattr(models, "pollinations_chat",
                        lambda *a, **k: {"content": _refuse_text("")})
    rc = probe_mod.main(["--model", "pollinations:openai"])
    assert rc == 1


def test_probe_one_model_unknown_provider_sets_error():
    # _probe_via_provider raises before the probe loop; probe_one_model
    # catches it as a provider-level failure, not a refusal.
    entry = probe_mod.probe_one_model("nosuch", "nosuch:m")
    assert entry["passed"] == 0 and entry["probes"] == []
    assert "unknown provider nosuch" in entry["error"]


def test_run_probe_text_empty_response():
    assert probe_mod._run_probe_text("p", lambda _p: "") == (
        "", "ERROR: empty response")
    assert probe_mod._run_probe_text("p", lambda _p: "   ") == (
        "", "ERROR: empty response")
    assert probe_mod._run_probe_text("p", lambda _p: None) == (
        "", "ERROR: empty response")


def test_probe_via_pollinations_error_recorded(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("pollinations down")
    monkeypatch.setattr(models, "pollinations_chat", boom)
    entry = probe_mod.probe_one_model("pollinations", "pollinations:openai")
    assert entry["passed"] == 0
    assert all(p["detail"] == "ERROR: pollinations down"
               for p in entry["probes"])
    assert "pollinations down" in entry["error"]


def test_probe_via_provider_error_recorded(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("leg down")
    monkeypatch.setattr(providers_mod, "call_provider", boom)
    entry = probe_mod.probe_one_model("llm7", "llm7:codestral-latest")
    assert entry["passed"] == 0
    assert all(p["detail"] == "ERROR: leg down" for p in entry["probes"])
    assert "leg down" in entry["error"]


def test_summarize_lists_skipped_models(stub_all_ok):
    report = probe_mod.run_suite()
    assert any(s["id"].startswith("groq:") for s in report["skipped"])
    assert "  [SKIP] groq:" in probe_mod.summarize(report)


def test_main_block_exits_without_probing(monkeypatch):
    # Covers `if __name__ == "__main__": sys.exit(main())`: re-execute
    # the module as __main__ with no models listed, so the suite probes
    # nothing and exits 1 — no network touched.
    import runpy
    import warnings
    monkeypatch.setattr(providers_mod, "list_models", lambda: [])
    monkeypatch.setattr(sys, "argv", ["probe"])
    with warnings.catch_warnings():
        # runpy warns that app.probe is already imported; expected here
        # — the re-execution is the behaviour under test.
        warnings.filterwarnings(
            "ignore", message=".*found in sys.modules.*",
            category=RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("app.probe", run_name="__main__")
    assert exc.value.code == 1
