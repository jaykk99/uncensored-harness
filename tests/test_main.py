"""Tests for app/main.py HTTP gates.

The harness is access-code gated by design: every endpoint that serves
agent work must require the code. /api/shot/{name} serves screenshots of
anything the agent did (browser sessions, files, chats), so it must not
be publicly fetchable.
"""

from __future__ import annotations

import os

from app import tools as T

CODE = "secret123"


def _client():
    from fastapi.testclient import TestClient
    from app import main as main_mod
    return TestClient(main_mod.app)


def test_shot_requires_access_code(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    c = _client()
    r = c.get("/api/shot/whatever.png")
    assert r.status_code == 401
    r = c.get("/api/shot/whatever.png?access_code=wrong")
    assert r.status_code == 401


def test_shot_wrong_code_is_401_not_404(monkeypatch):
    # Auth is checked before the file lookup: a wrong code must not leak
    # whether the screenshot exists.
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    c = _client()
    r = c.get("/api/shot/definitely-not-here.png?access_code=nope")
    assert r.status_code == 401


def test_shot_serves_with_code(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    p = os.path.join(T.SHOTS_DIR, "test-shot-gate.png")
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nfakepng")
    try:
        c = _client()
        r = c.get(f"/api/shot/test-shot-gate.png?access_code={CODE}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content.startswith(b"\x89PNG")
    finally:
        os.remove(p)


def test_shot_missing_file_is_404_with_code(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    c = _client()
    r = c.get(f"/api/shot/no-such-shot.png?access_code={CODE}")
    assert r.status_code == 404


def test_router_stats_reset_requires_code(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    c = _client()
    r = c.post("/api/router-stats/reset", json={})
    assert r.status_code == 401
    r = c.post("/api/router-stats/reset",
               json={"access_code": "wrong"})
    assert r.status_code == 401


def test_router_stats_reset_clears(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    from app import router_stats as rs_mod
    rs_mod.record("test-provider", "test-model", "error", 10)
    assert any(l["provider"] == "test-provider" for l in rs_mod.snapshot())
    c = _client()
    r = c.post("/api/router-stats/reset",
               json={"access_code": CODE, "provider": "test-provider"})
    assert r.status_code == 200
    assert r.json()["cleared"] == 1
    assert not any(l["provider"] == "test-provider"
                   for l in rs_mod.snapshot())


def test_health_does_not_leak_workdir(monkeypatch):
    # /api/health is unauthenticated by design (liveness); it must not
    # disclose server filesystem paths.
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    c = _client()
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "workdir" not in body
    assert body.get("ok") is True


def test_probe_rejects_unknown_provider(monkeypatch):
    # A typo'd provider must 400, not silently probe zero models and
    # report "0/0 models fully clean".
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    from app import probe as probe_mod
    def _must_not_run(**kwargs):
        raise AssertionError("run_suite must not be called for a bad provider")
    monkeypatch.setattr(probe_mod, "run_suite", _must_not_run)
    c = _client()
    r = c.get(f"/api/probe?provider=definitely-not-a-provider&access_code={CODE}")
    assert r.status_code == 400
    assert "unknown provider" in r.json()["detail"]


def test_probe_passes_known_provider_to_suite(monkeypatch):
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    from app import probe as probe_mod
    seen = {}
    def fake_suite(provider="", model_id="", timeout_s=120):
        seen.update(provider=provider, model_id=model_id)
        return {"ok": True, "probed": 0, "passed_models": 0, "skipped": [],
                "verdict": "ok", "results": []}
    monkeypatch.setattr(probe_mod, "run_suite", fake_suite)
    c = _client()
    r = c.get(f"/api/probe?provider=llm7&access_code={CODE}")
    assert r.status_code == 200
    assert seen == {"provider": "llm7", "model_id": ""}
    # Virtual providers from the model picker resolve too.
    seen.clear()
    r = c.get(f"/api/probe?provider=horde&access_code={CODE}")
    assert r.status_code == 200
    assert seen["provider"] == "horde"
