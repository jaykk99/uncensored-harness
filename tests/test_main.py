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


def _thread_client(monkeypatch, tmp_path):
    """TestClient with thread storage pointed at a tmp db."""
    from app import memory
    monkeypatch.setattr(memory, "DB", str(tmp_path / "threads.db"))
    monkeypatch.setenv("HARNESS_ACCESS_CODE", CODE)
    from app import memory as mem_mod
    tid = mem_mod.create_thread("reopen me")
    mem_mod.add_message(tid, "user", "first question")
    mem_mod.add_message(tid, "assistant", "first answer")
    return _client(), tid


def test_thread_detail_requires_code(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}")
    assert r.status_code == 401
    r = c.get(f"/api/threads/{tid}?access_code=wrong")
    assert r.status_code == 401


def test_thread_detail_serves_history_in_order(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}?access_code={CODE}")
    assert r.status_code == 200
    body = r.json()
    assert body["thread"]["id"] == tid
    assert body["thread"]["title"] == "reopen me"
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "first question"), ("assistant", "first answer")]


def test_thread_detail_limit_clamped(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}?access_code={CODE}&limit=1")
    assert len(r.json()["messages"]) == 1
    assert r.json()["messages"][0]["content"] == "first answer"


def test_thread_detail_offset_pages_and_reports_total(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"extra-{i}")
    body = c.get(f"/api/threads/{tid}?access_code={CODE}").json()
    assert body["total"] == 5
    assert [m["content"] for m in body["messages"]] == [
        "first question", "first answer", "extra-0", "extra-1", "extra-2"]
    older = c.get(
        f"/api/threads/{tid}?access_code={CODE}&limit=2&offset=2").json()
    assert older["total"] == 5
    assert [m["content"] for m in older["messages"]] == [
        "first answer", "extra-0"]
    assert c.get(
        f"/api/threads/{tid}?access_code={CODE}&offset=99").json()["messages"] == []
    # Negative offset clamps to the newest page, like limit does.
    newest = c.get(
        f"/api/threads/{tid}?access_code={CODE}&offset=-3").json()
    assert [m["content"] for m in newest["messages"]][-1] == "extra-2"


def test_thread_detail_unknown_is_404_not_auth_leak(monkeypatch, tmp_path):
    c, _tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/nope-not-real?access_code={CODE}")
    assert r.status_code == 404
    # Wrong code still 401s first: existence is not leaked to strangers.
    r = c.get("/api/threads/nope-not-real?access_code=wrong")
    assert r.status_code == 401


def test_thread_delete_removes_and_404s_after(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.delete(f"/api/threads/{tid}?access_code=wrong")
    assert r.status_code == 401
    r = c.delete(f"/api/threads/nope-not-real?access_code={CODE}")
    assert r.status_code == 404
    r = c.delete(f"/api/threads/{tid}?access_code={CODE}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    r = c.get(f"/api/threads/{tid}?access_code={CODE}")
    assert r.status_code == 404


def test_thread_rename_updates_list(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.patch(f"/api/threads/{tid}", json={"title": "  ", "access_code": CODE})
    assert r.status_code == 400
    r = c.patch("/api/threads/nope-not-real",
                json={"title": "x", "access_code": CODE})
    assert r.status_code == 404
    r = c.patch(f"/api/threads/{tid}",
                json={"title": "renamed thread", "access_code": CODE})
    assert r.status_code == 200
    assert r.json()["thread"]["title"] == "renamed thread"
    r = c.get(f"/api/threads?access_code={CODE}")
    titles = [t["title"] for t in r.json()["threads"]]
    assert "renamed thread" in titles


def test_thread_duplicate_copies_and_opens_copy(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/duplicate",
               json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/threads/nope-not-real/duplicate",
               json={"access_code": CODE})
    assert r.status_code == 404
    r = c.post(f"/api/threads/{tid}/duplicate", json={"access_code": CODE})
    assert r.status_code == 200
    body = r.json()
    assert body["duplicated"] is True
    assert body["messages"] == 2
    new_id = body["thread"]["id"]
    assert new_id != tid
    assert body["thread"]["title"] == "reopen me (copy)"
    # Copy carries the full history; original is untouched.
    copy = c.get(f"/api/threads/{new_id}?access_code={CODE}").json()
    assert [(m["role"], m["content"]) for m in copy["messages"]] == [
        ("user", "first question"), ("assistant", "first answer")]
    orig = c.get(f"/api/threads/{tid}?access_code={CODE}").json()
    assert orig["total"] == 2
    titles = [t["title"] for t in
              c.get(f"/api/threads?access_code={CODE}").json()["threads"]]
    assert "reopen me (copy)" in titles


def test_thread_pin_toggles_sets_and_sorts_first(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    newer = memory.create_thread("newer chat")
    # Auth first, then 404 — same gate order as the other thread endpoints.
    r = c.post(f"/api/threads/{tid}/pin", json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/threads/nope-not-real/pin", json={"access_code": CODE})
    assert r.status_code == 404
    # Toggle (no pinned value): off -> on, and the pinned thread jumps
    # above the newer unpinned one in the sidebar list.
    r = c.post(f"/api/threads/{tid}/pin", json={"access_code": CODE})
    assert r.status_code == 200
    assert r.json()["pinned"] is True
    assert r.json()["thread"]["pinned"] is True
    listed = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert [t["id"] for t in listed] == [tid, newer]
    # Explicit set back to unpinned restores newest-first order.
    r = c.post(f"/api/threads/{tid}/pin",
               json={"pinned": False, "access_code": CODE})
    assert r.json()["pinned"] is False
    listed = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert [t["id"] for t in listed] == [newer, tid]


def test_thread_archive_toggles_sets_and_hides_from_list(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("still active")
    # Auth first, then 404 — same gate order as the other thread endpoints.
    r = c.post(f"/api/threads/{tid}/archive", json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/threads/nope-not-real/archive", json={"access_code": CODE})
    assert r.status_code == 404
    # Toggle (no archived value): off -> on, and the thread leaves the
    # sidebar list but shows in the archive view.
    r = c.post(f"/api/threads/{tid}/archive", json={"access_code": CODE})
    assert r.status_code == 200
    assert r.json()["archived"] is True
    assert r.json()["thread"]["archived"] is True
    listed = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert [t["id"] for t in listed] == [other]
    archived = c.get(f"/api/threads?archived=true&access_code={CODE}").json()["threads"]
    assert [t["id"] for t in archived] == [tid]
    # Archiving never deletes: detail and export still serve it.
    assert c.get(f"/api/threads/{tid}?access_code={CODE}").status_code == 200
    assert c.get(f"/api/threads/{tid}/export?access_code={CODE}").status_code == 200
    # Explicit set back to unarchived restores it to the sidebar list.
    r = c.post(f"/api/threads/{tid}/archive",
               json={"archived": False, "access_code": CODE})
    assert r.json()["archived"] is False
    listed = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert [t["id"] for t in listed] == [other, tid]


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


def test_thread_search_requires_code(monkeypatch, tmp_path):
    c, _tid = _thread_client(monkeypatch, tmp_path)
    r = c.get("/api/threads/search?q=first")
    assert r.status_code == 401
    r = c.get("/api/threads/search?q=first&access_code=wrong")
    assert r.status_code == 401


def test_thread_search_finds_by_content_and_title(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/search?q=first+question&access_code={CODE}")
    assert r.status_code == 200
    hits = r.json()["threads"]
    assert [h["id"] for h in hits] == [tid]
    assert "first question" in hits[0]["snippet"]
    r = c.get(f"/api/threads/search?q=reopen&access_code={CODE}")
    assert [h["id"] for h in r.json()["threads"]] == [tid]
    r = c.get(f"/api/threads/search?q=zzz-no-match&access_code={CODE}")
    assert r.json()["threads"] == []


def test_thread_search_hit_carries_pinned_flag(monkeypatch, tmp_path):
    # Search hits render through the same sidebar list as /api/threads,
    # so a pinned hit must carry the flag or its star renders unpinned.
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/pin", json={"access_code": CODE})
    assert r.json()["pinned"] is True
    r = c.get(f"/api/threads/search?q=first+question&access_code={CODE}")
    hits = r.json()["threads"]
    assert [h["id"] for h in hits] == [tid]
    assert hits[0]["pinned"] is True


def test_thread_search_route_is_not_a_thread_id(monkeypatch, tmp_path):
    # "search" must hit the search endpoint, not thread_detail's 404.
    c, _tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/search?q=first&access_code={CODE}")
    assert r.status_code == 200


def test_thread_export_markdown_download(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}/export?access_code={CODE}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert f'filename="thread-{tid}.md"' in r.headers["content-disposition"]
    body = r.text
    assert body.startswith("# reopen me\n")
    assert body.index("first question") < body.index("first answer")
    assert "## You" in body and "## Assistant" in body


def test_thread_export_json_and_full_thread(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    # Past the UI's 60-message page: an export must still carry all of it.
    for i in range(65):
        memory.add_message(tid, "user", f"bulk-{i}")
    r = c.get(f"/api/threads/{tid}/export?format=json&access_code={CODE}")
    assert r.status_code == 200
    assert f'filename="thread-{tid}.json"' in r.headers["content-disposition"]
    body = r.json()
    assert body["thread"]["id"] == tid
    assert body["total"] == 67
    assert body["messages"][0]["content"] == "first question"
    assert body["messages"][-1]["content"] == "bulk-64"


def test_thread_export_guards(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}/export")
    assert r.status_code == 401
    r = c.get(f"/api/threads/{tid}/export?access_code=wrong")
    assert r.status_code == 401
    r = c.get(f"/api/threads/nope-not-real/export?access_code={CODE}")
    assert r.status_code == 404
    r = c.get(f"/api/threads/{tid}/export?format=pdf&access_code={CODE}")
    assert r.status_code == 400
