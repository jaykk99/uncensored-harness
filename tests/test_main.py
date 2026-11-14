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


def test_thread_detail_messages_carry_ids(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    body = c.get(f"/api/threads/{tid}?access_code={CODE}").json()
    assert all("id" in m for m in body["messages"])
    assert [m["content"] for m in body["messages"]] == [
        "first question", "first answer"]


def _prune_client(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for role, content in [("user", "q1"), ("assistant", "a1"),
                          ("user", "q2"), ("assistant", "a2")]:
        memory.add_message(tid, role, content)
    # _thread_client seeds "first question"/"first answer" first; the
    # prune targets are the four messages added above, in order.
    all_msgs = c.get(
        f"/api/threads/{tid}?access_code={CODE}").json()["messages"]
    ids = [m["id"] for m in all_msgs
           if m["content"] in ("q1", "a1", "q2", "a2")]
    assert [m["content"] for m in all_msgs][-4:] == ["q1", "a1", "q2", "a2"]
    return c, tid, ids


def test_thread_message_delete_prunes_one(monkeypatch, tmp_path):
    c, tid, ids = _prune_client(monkeypatch, tmp_path)
    r = c.delete(f"/api/threads/{tid}/messages/{ids[1]}?access_code=wrong")
    assert r.status_code == 401
    r = c.delete(f"/api/threads/nope-not-real/messages/{ids[1]}?access_code={CODE}")
    assert r.status_code == 404
    r = c.delete(f"/api/threads/{tid}/messages/999999?access_code={CODE}")
    assert r.status_code == 404
    r = c.delete(f"/api/threads/{tid}/messages/{ids[1]}?access_code={CODE}")
    assert r.status_code == 200
    assert r.json()["remaining"] == 5
    body = c.get(f"/api/threads/{tid}?access_code={CODE}").json()
    assert [m["content"] for m in body["messages"]] == [
        "first question", "first answer", "q1", "q2", "a2"]


def test_thread_message_truncate_cuts_tail(monkeypatch, tmp_path):
    c, tid, ids = _prune_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/messages/{ids[2]}/truncate",
               json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post(f"/api/threads/{tid}/messages/999999/truncate",
               json={"access_code": CODE})
    assert r.status_code == 404
    r = c.post(f"/api/threads/{tid}/messages/{ids[2]}/truncate",
               json={"access_code": CODE})
    assert r.status_code == 200
    assert r.json()["deleted"] == 2
    assert r.json()["remaining"] == 4
    body = c.get(f"/api/threads/{tid}?access_code={CODE}").json()
    assert [m["content"] for m in body["messages"]] == [
        "first question", "first answer", "q1", "a1"]


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


def test_thread_import_roundtrips_export(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    # Export the thread, then import its JSON: the restored thread is a
    # new id with the same title and messages, visible in the list.
    exported = c.get(f"/api/threads/{tid}/export?format=json&access_code={CODE}")
    assert exported.status_code == 200
    payload = exported.json()
    payload["access_code"] = CODE
    r = c.post("/api/threads/import", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] is True
    assert body["messages"] == 2
    new_id = body["thread"]["id"]
    assert new_id != tid
    assert body["thread"]["title"] == "reopen me"
    assert body["thread"]["pinned"] is False
    assert body["thread"]["archived"] is False
    restored = c.get(f"/api/threads/{new_id}?access_code={CODE}").json()
    assert [(m["role"], m["content"]) for m in restored["messages"]] == [
        ("user", "first question"), ("assistant", "first answer")]
    titles = [t["title"] for t in
              c.get(f"/api/threads?access_code={CODE}").json()["threads"]]
    assert titles.count("reopen me") == 2


def test_thread_import_requires_code_and_validates_payload(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    good = {"thread": {"title": "x"},
            "messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/api/threads/import", json={**good, "access_code": "wrong"})
    assert r.status_code == 401
    # Auth passes, payload fails: nothing is imported.
    r = c.post("/api/threads/import",
               json={"thread": {}, "messages": "not a list",
                     "access_code": CODE})
    assert r.status_code == 400
    r = c.post("/api/threads/import",
               json={"thread": {},
                     "messages": [{"role": "user"}],
                     "access_code": CODE})
    assert r.status_code == 400
    r = c.post("/api/threads/import",
               json={"thread": {},
                     "messages": [{"role": "user", "content": 42}],
                     "access_code": CODE})
    assert r.status_code == 400
    before = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    r = c.post("/api/threads/import", json={**good, "access_code": CODE})
    assert r.status_code == 200
    assert r.json()["thread"]["title"] == "x"
    after = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert len(after) == len(before) + 1


def test_threads_export_all_json_covers_every_thread(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("second chat")
    memory.add_message(other, "user", "hello there")
    archived = memory.create_thread("old archived chat")
    memory.add_message(archived, "user", "archived words")
    memory.set_thread_archived(archived, True)
    r = c.get("/api/threads/export-all")
    assert r.status_code == 401
    r = c.get(f"/api/threads/export-all?access_code=wrong")
    assert r.status_code == 401
    r = c.get(f"/api/threads/export-all?format=txt&access_code={CODE}")
    assert r.status_code == 400
    r = c.get(f"/api/threads/export-all?format=json&access_code={CODE}")
    assert r.status_code == 200
    assert "harness-backup-" in r.headers["content-disposition"]
    body = r.json()
    # Archived threads are in the backup too: a backup that skips
    # them is not a backup.
    assert body["total_threads"] == 3
    assert body["total_messages"] == 4
    titles = [e["thread"]["title"] for e in body["threads"]]
    assert titles == ["old archived chat", "second chat", "reopen me"]
    by_title = {e["thread"]["title"]: e["messages"] for e in body["threads"]}
    assert [(m["role"], m["content"]) for m in by_title["reopen me"]] == [
        ("user", "first question"), ("assistant", "first answer")]
    assert by_title["old archived chat"][0]["content"] == "archived words"


def test_threads_export_all_past_the_50_cap(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(54):
        memory.create_thread(f"bulk-{i}")
    body = c.get(f"/api/threads/export-all?access_code={CODE}").json()
    assert body["total_threads"] == 55
    assert len(body["threads"]) == 55


def test_threads_export_all_markdown(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/export-all?format=md&access_code={CODE}")
    assert r.status_code == 200
    assert "harness-backup-" in r.headers["content-disposition"]
    assert "# reopen me" in r.text
    assert "first question" in r.text
    assert "first answer" in r.text


def test_thread_import_bundle_roundtrips_export_all(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("second chat")
    memory.add_message(other, "user", "hello there")
    exported = c.get(f"/api/threads/export-all?access_code={CODE}")
    assert exported.status_code == 200
    payload = exported.json()
    payload["access_code"] = CODE
    r = c.post("/api/threads/import", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] is True
    assert body["threads"] == 2
    assert body["messages"] == 3
    assert len(body["thread_ids"]) == 2
    titles = [t["title"] for t in
              c.get(f"/api/threads?access_code={CODE}").json()["threads"]]
    assert titles.count("reopen me") == 2
    assert titles.count("second chat") == 2
    # Export-all lists newest first, so thread_ids[1] is the restored
    # "reopen me" (thread_ids[0] is "second chat").
    restored = c.get(
        f"/api/threads/{body['thread_ids'][1]}?access_code={CODE}").json()
    assert [(m["role"], m["content"]) for m in restored["messages"]] == [
        ("user", "first question"), ("assistant", "first answer")]


def test_thread_import_bundle_validates_everything_first(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    good = {"thread": {"title": "ok"},
            "messages": [{"role": "user", "content": "hi"}]}
    bad = {"thread": {"title": "bad"},
           "messages": [{"role": "user", "content": 42}]}
    before = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    r = c.post("/api/threads/import",
               json={"threads": "not a list", "access_code": CODE})
    assert r.status_code == 400
    # One bad entry: nothing is imported, not even the good entry.
    r = c.post("/api/threads/import",
               json={"threads": [good, bad], "access_code": CODE})
    assert r.status_code == 400
    after = c.get(f"/api/threads?access_code={CODE}").json()["threads"]
    assert len(after) == len(before)


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


def test_thread_message_search_requires_code_and_thread(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.get(f"/api/threads/{tid}/messages/search?q=first")
    assert r.status_code == 401
    r = c.get(f"/api/threads/{tid}/messages/search?q=first&access_code=wrong")
    assert r.status_code == 401
    r = c.get(f"/api/threads/nope-not-real/messages/search?q=first&access_code={CODE}")
    assert r.status_code == 404


def test_thread_message_search_finds_within_thread_only(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("other")
    memory.add_message(other, "user", "needle elsewhere")
    memory.add_message(tid, "assistant", "a needle answer here")
    r = c.get(f"/api/threads/{tid}/messages/search?q=needle&access_code={CODE}")
    assert r.status_code == 200
    body = r.json()
    # Only this thread's match, at thread position 3 — the other
    # thread's match never leaks in.
    assert body["total"] == 1
    assert body["thread_total"] == 3
    assert [m["position"] for m in body["matches"]] == [3]
    assert [m["role"] for m in body["matches"]] == ["assistant"]
    assert all("needle" in m["snippet"].lower() for m in body["matches"])
    r = c.get(f"/api/threads/{tid}/messages/search?q=zzz-no-match&access_code={CODE}")
    assert r.json()["matches"] == [] and r.json()["total"] == 0
    r = c.get(f"/api/threads/{tid}/messages/search?access_code={CODE}")
    assert r.json()["matches"] == []


def test_thread_message_search_limit_clamped(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"commonword {i}")
    body = c.get(
        f"/api/threads/{tid}/messages/search?q=commonword&limit=2&access_code={CODE}").json()
    assert body["total"] == 3 and len(body["matches"]) == 2
    body = c.get(
        f"/api/threads/{tid}/messages/search?q=commonword&limit=999&access_code={CODE}").json()
    assert len(body["matches"]) == 3  # clamped to 50, not 999


def test_global_message_search_requires_code(monkeypatch, tmp_path):
    c, _tid = _thread_client(monkeypatch, tmp_path)
    r = c.get("/api/messages/search?q=first")
    assert r.status_code == 401
    r = c.get("/api/messages/search?q=first&access_code=wrong")
    assert r.status_code == 401


def test_global_message_search_finds_turns_across_threads(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("other chat")
    memory.add_message(other, "user", "needle elsewhere")
    memory.add_message(tid, "assistant", "a needle answer here")
    r = c.get(f"/api/messages/search?q=needle&access_code={CODE}")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    # Newest match first: the answer just added, then the other thread.
    assert [m["thread_id"] for m in body["matches"]] == [tid, other]
    # The hit carries the address the UI jumps with: row id, thread
    # position, thread total, thread title, and role.
    hit = body["matches"][0]
    assert hit["position"] == 3 and hit["thread_total"] == 3
    assert hit["thread_title"] == "reopen me" and hit["role"] == "assistant"
    assert "needle" in hit["snippet"].lower()
    r = c.get(f"/api/messages/search?q=zzz-no-match&access_code={CODE}")
    assert r.json()["matches"] == [] and r.json()["total"] == 0
    r = c.get(f"/api/messages/search?access_code={CODE}")
    assert r.json()["matches"] == []


def test_global_message_search_limit_clamped(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"commonword {i}")
    body = c.get(
        f"/api/messages/search?q=commonword&limit=2&access_code={CODE}").json()
    assert body["total"] == 3 and len(body["matches"]) == 2
    body = c.get(
        f"/api/messages/search?q=commonword&limit=999&access_code={CODE}").json()
    assert len(body["matches"]) == 3  # clamped to 50, not 999


def test_thread_message_search_pages_with_after_id(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"commonword {i}")
    p1 = c.get(
        f"/api/threads/{tid}/messages/search?q=commonword&limit=2&access_code={CODE}").json()
    assert p1["total"] == 3 and len(p1["matches"]) == 2
    assert p1["has_more"] is True and p1["next_after_id"]
    p2 = c.get(
        f"/api/threads/{tid}/messages/search?q=commonword&limit=2"
        f"&after_id={p1['next_after_id']}&access_code={CODE}").json()
    assert len(p2["matches"]) == 1 and p2["has_more"] is False
    assert p2["next_after_id"] is None
    assert p1["matches"][0]["id"] != p2["matches"][0]["id"]


def test_global_message_search_pages_with_before_id(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    other = memory.create_thread("other chat")
    for i in range(2):
        memory.add_message(other, "user", f"commonword {i}")
    for i in range(2):
        memory.add_message(tid, "user", f"commonword {i}")
    p1 = c.get(
        f"/api/messages/search?q=commonword&limit=2&access_code={CODE}").json()
    assert p1["total"] == 4 and len(p1["matches"]) == 2
    assert p1["has_more"] is True and p1["next_before_id"]
    p2 = c.get(
        f"/api/messages/search?q=commonword&limit=2"
        f"&before_id={p1['next_before_id']}&access_code={CODE}").json()
    assert len(p2["matches"]) == 2 and p2["has_more"] is False
    assert p2["next_before_id"] is None
    seen = [m["id"] for m in p1["matches"] + p2["matches"]]
    assert len(seen) == len(set(seen)) == 4
    # The cursor is still behind the access-code gate.
    r = c.get(
        f"/api/messages/search?q=commonword&limit=2"
        f"&before_id={p1['next_before_id']}")
    assert r.status_code == 401


def test_thread_message_position_endpoint(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"extra-{i}")
    ids = [m["id"] for m in c.get(
        f"/api/threads/{tid}?access_code={CODE}").json()["messages"]]
    # _thread_client seeds 2 messages, then 3 extras: positions 1-5.
    r = c.get(f"/api/threads/{tid}/messages/{ids[0]}/position?access_code={CODE}")
    assert r.status_code == 200
    assert r.json() == {"thread_id": tid, "message_id": ids[0],
                        "position": 1, "total": 5}
    r = c.get(f"/api/threads/{tid}/messages/{ids[-1]}/position?access_code={CODE}")
    assert r.json()["position"] == 5
    # Auth first, then thread, then the row's membership in this
    # thread: none of them may leak the others.
    r = c.get(f"/api/threads/{tid}/messages/{ids[0]}/position")
    assert r.status_code == 401
    r = c.get(f"/api/threads/{tid}/messages/{ids[0]}/position?access_code=wrong")
    assert r.status_code == 401
    r = c.get(f"/api/threads/nope-not-real/messages/{ids[0]}/position?access_code={CODE}")
    assert r.status_code == 404
    r = c.get(f"/api/threads/{tid}/messages/999999/position?access_code={CODE}")
    assert r.status_code == 404
    other = memory.create_thread("other")
    memory.add_message(other, "user", "elsewhere")
    foreign = c.get(
        f"/api/threads/{other}?access_code={CODE}").json()["messages"][0]["id"]
    r = c.get(f"/api/threads/{tid}/messages/{foreign}/position?access_code={CODE}")
    assert r.status_code == 404


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


def test_thread_detail_before_id_pages_and_reports_total(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    for i in range(3):
        memory.add_message(tid, "user", f"extra-{i}")
    body = c.get(f"/api/threads/{tid}?access_code={CODE}&limit=2").json()
    assert body["total"] == 5
    assert [m["content"] for m in body["messages"]] == ["extra-1", "extra-2"]
    cursor = body["messages"][0]["id"]
    older = c.get(
        f"/api/threads/{tid}?access_code={CODE}&limit=2&before_id={cursor}").json()
    assert older["total"] == 5
    assert [m["content"] for m in older["messages"]] == ["first answer", "extra-0"]
    first_id = older["messages"][0]["id"]
    oldest = c.get(
        f"/api/threads/{tid}?access_code={CODE}&limit=2&before_id={first_id}").json()
    assert [m["content"] for m in oldest["messages"]] == ["first question"]
    # Cursor beats offset: offset=0 alone would return the newest page.
    again = c.get(
        f"/api/threads/{tid}?access_code={CODE}&limit=2&offset=0"
        f"&before_id={cursor}").json()
    assert [m["content"] for m in again["messages"]] == ["first answer", "extra-0"]
    # Auth still checked before the lookup, like the other thread routes.
    r = c.get(f"/api/threads/{tid}?access_code=wrong&before_id={cursor}")
    assert r.status_code == 401
    r = c.get(f"/api/threads/nope-not-real?access_code={CODE}&before_id={cursor}")
    assert r.status_code == 404


def _fake_turn(monkeypatch, reply="fresh answer"):
    """Stand in for agent.run_turn: emit one text event + done, return
    the reply. Records the history it was fed so tests can assert the
    stale assistant reply was not in it."""
    from app import agent as agent_mod
    seen = {}

    async def fake(history, emit, model="", web_search=False,
                 should_stop=None, extra_instructions=""):
        seen["history"] = list(history)
        seen["model"] = model
        seen["web_search"] = web_search
        seen["extra_instructions"] = extra_instructions
        await emit({"type": "text", "text": reply})
        await emit({"type": "done", "model": "fake-model"})
        return reply

    monkeypatch.setattr(agent_mod, "run_turn", fake)
    return seen


def test_regenerate_requires_code_and_thread(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/regenerate",
               json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/threads/nope-not-real/regenerate",
               json={"access_code": CODE})
    assert r.status_code == 404


def test_regenerate_rejects_empty_and_promptless_threads(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    empty = memory.create_thread("empty")
    r = c.post(f"/api/threads/{empty}/regenerate",
               json={"access_code": CODE})
    assert r.status_code == 400
    promptless = memory.create_thread("promptless")
    memory.add_message(promptless, "assistant", "orphan reply")
    r = c.post(f"/api/threads/{promptless}/regenerate",
               json={"access_code": CODE})
    assert r.status_code == 400


def test_regenerate_drops_old_reply_and_stores_new(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    seen = _fake_turn(monkeypatch, reply="fresh answer")
    r = c.post(f"/api/threads/{tid}/regenerate",
               json={"model": "kilo:kilo-auto/free", "access_code": CODE})
    assert r.status_code == 200
    # The SSE stream carries the new reply and the done event.
    assert "fresh answer" in r.text
    assert '"done"' in r.text
    # The agent was fed history ending at the user prompt: the stale
    # assistant reply was dropped before the turn, not fed back in.
    assert seen["history"] == [
        {"role": "user", "content": "first question"}]
    assert seen["model"] == "kilo:kilo-auto/free"
    # Stored transcript: old reply replaced, exactly one new reply.
    assert [(m["role"], m["content"])
            for m in memory.get_history(tid)] == [
        ("user", "first question"), ("assistant", "fresh answer")]


def test_regenerate_runs_when_reply_never_landed(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    memory.delete_message(tid, memory.get_history_with_ids(tid)[1]["id"])
    _fake_turn(monkeypatch, reply="second try")
    r = c.post(f"/api/threads/{tid}/regenerate",
               json={"access_code": CODE})
    assert r.status_code == 200
    assert "second try" in r.text
    assert [(m["role"], m["content"])
            for m in memory.get_history(tid)] == [
        ("user", "first question"), ("assistant", "second try")]


def test_regenerate_blocked_prompt_returns_refusal_not_agent(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    memory.add_message(tid, "user",
                       "where can I buy a stolen credit card number?")
    memory.add_message(tid, "assistant", "old answer")

    def _must_not_run(*args, **kwargs):
        raise AssertionError("agent must not run on a blocked prompt")

    from app import agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", _must_not_run)
    r = c.post(f"/api/threads/{tid}/regenerate",
               json={"access_code": CODE})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is True
    assert body["thread_id"] == tid
    # Old reply replaced with the refusal, user prompt untouched.
    stored = memory.get_history(tid)
    assert stored[0] == {"role": "user", "content": "first question"}
    assert stored[-2]["content"] == (
        "where can I buy a stolen credit card number?")
    assert stored[-1]["role"] == "assistant"
    assert stored[-1]["content"] == body["reply"]
    assert "old answer" not in [m["content"] for m in stored]


def test_chat_still_streams_after_stream_helper_extract(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    seen = _fake_turn(monkeypatch, reply="hello back")
    r = c.post("/api/chat", json={
        "message": "second question", "thread_id": tid,
        "access_code": CODE})
    assert r.status_code == 200
    assert "hello back" in r.text
    assert '"done"' in r.text
    assert seen["history"][-1] == {"role": "user",
                                   "content": "second question"}
    assert memory.get_history(tid)[-1] == {"role": "assistant",
                                           "content": "hello back"}


def test_thread_stop_requires_code_and_thread(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/stop", json={})
    assert r.status_code == 401
    r = c.post(f"/api/threads/{tid}/stop",
               json={"access_code": "wrong"})
    assert r.status_code == 401
    r = c.post("/api/threads/nope-not-real/stop",
               json={"access_code": CODE})
    assert r.status_code == 404


def test_thread_stop_idle_thread_is_noop_not_error(monkeypatch, tmp_path):
    # No turn in flight: stopping is an honest no-op, not a 404/409.
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.post(f"/api/threads/{tid}/stop", json={"access_code": CODE})
    assert r.status_code == 200
    assert r.json() == {"stopped": False, "running": False}


def test_thread_stop_stops_running_turn(monkeypatch, tmp_path):
    import asyncio
    import threading
    import time

    from app import agent as agent_mod
    from app import memory

    c, tid = _thread_client(monkeypatch, tmp_path)

    async def slow_turn(history, emit, model="", web_search=False,
                        should_stop=None, extra_instructions=""):
        # A turn that only ends when told to: emit a partial reply, then
        # poll the stop flag the way the real agent loop checks it.
        await emit({"type": "text", "text": "partial answer"})
        for _ in range(200):
            if should_stop and should_stop():
                await emit({"type": "stopped"})
                return "partial answer\n[stopped by user]"
            await asyncio.sleep(0.05)
        return "partial answer"

    monkeypatch.setattr(agent_mod, "run_turn", slow_turn)
    result = {}

    def _chat():
        result["resp"] = c.post("/api/chat", json={
            "message": "second question", "thread_id": tid,
            "access_code": CODE})

    t = threading.Thread(target=_chat, daemon=True)
    t.start()
    # Give the turn time to start streaming before stopping it.
    for _ in range(100):
        time.sleep(0.05)
        probe = c.post(f"/api/threads/{tid}/stop",
                       json={"access_code": CODE})
        if probe.json().get("running"):
            break
    else:
        raise AssertionError("turn never registered as running")
    assert probe.status_code == 200
    assert probe.json() == {"stopped": True, "running": True}
    t.join(timeout=15)
    assert not t.is_alive()
    r = result["resp"]
    assert r.status_code == 200
    assert "partial answer" in r.text
    assert '"stopped"' in r.text
    # The stored reply is the partial answer with the marker — not a
    # full reply, and not missing (the turn did run).
    stored = memory.get_history(tid)
    assert stored[-1] == {"role": "assistant",
                          "content": "partial answer\n[stopped by user]"}
    # The flag is dropped when the turn ends: a later stop is a no-op.
    r = c.post(f"/api/threads/{tid}/stop", json={"access_code": CODE})
    assert r.json() == {"stopped": False, "running": False}


def test_run_turn_stop_before_first_model_call(monkeypatch):
    # Stopping before anything runs must not call the model at all.
    from app import agent as agent_mod
    from app import models
    from tests.conftest import run

    def _must_not_run(*args, **kwargs):
        raise AssertionError("model must not run on a stopped turn")

    monkeypatch.setattr(models, "chat", _must_not_run)
    seen = []

    async def emit(ev):
        seen.append(ev)

    result = run(agent_mod.run_turn(
        [{"role": "user", "content": "hi"}], emit,
        should_stop=lambda: True))
    assert result == "[stopped by user]"
    assert {"type": "stopped"} in seen
    assert seen[-1]["type"] == "done"


def test_run_turn_stop_skips_remaining_tool_calls(monkeypatch):
    # One tool call runs, then the stop lands: the second call never
    # executes and the next model call never happens.
    from app import agent as agent_mod
    from app import models
    from app import tools as T
    from tests.conftest import run

    calls = {"chat": 0, "tool": 0}
    stop_after_first_tool = {"flag": False}

    def _fake_chat(messages, max_tokens, temperature, tools,
                   progress=None, preferred_model="", on_token=None):
        calls["chat"] += 1
        text = ('```tool_call\n{"name": "echo_tool", "arguments": {}}\n```'
                if calls["chat"] == 1 else "second answer")
        return text, "fake", [], None, False

    def _echo_tool(**kwargs):
        calls["tool"] += 1
        stop_after_first_tool["flag"] = True
        return "tool output"

    monkeypatch.setattr(models, "chat", _fake_chat)
    monkeypatch.setattr(T, "get_tool",
                        lambda name: {"execute": _echo_tool})

    seen = []

    async def emit(ev):
        seen.append(ev)

    result = run(agent_mod.run_turn(
        [{"role": "user", "content": "do two things"}], emit,
        should_stop=lambda: stop_after_first_tool["flag"]))
    assert calls == {"chat": 1, "tool": 1}
    assert result == "[stopped by user]"
    assert {"type": "stopped"} in seen


def _edit_url(tid, mid):
    return f"/api/threads/{tid}/messages/{mid}/edit"


def test_message_edit_requires_code_and_thread(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    mid = memory.get_history_with_ids(tid)[0]["id"]
    r = c.post(_edit_url(tid, mid),
               json={"content": "x", "access_code": "wrong"})
    assert r.status_code == 401
    r = c.post(_edit_url("nope-not-real", mid),
               json={"content": "x", "access_code": CODE})
    assert r.status_code == 404
    r = c.post(_edit_url(tid, 999999),
               json={"content": "x", "access_code": CODE})
    assert r.status_code == 404
    # A message id from another thread never edits here.
    other = memory.create_thread("other")
    memory.add_message(other, "user", "elsewhere")
    foreign = c.get(
        f"/api/threads/{other}?access_code={CODE}").json()["messages"][0]["id"]
    r = c.post(_edit_url(tid, foreign),
               json={"content": "x", "access_code": CODE})
    assert r.status_code == 404


def test_message_edit_rejects_empty_content(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    mid = memory.get_history_with_ids(tid)[0]["id"]
    r = c.post(_edit_url(tid, mid),
               json={"content": "   ", "access_code": CODE})
    assert r.status_code == 400
    assert memory.get_history(tid)[0]["content"] == "first question"


def test_message_edit_in_place_updates_stored_text(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    ids = [m["id"] for m in memory.get_history_with_ids(tid)]
    # Either role can be corrected in place; nothing else moves.
    r = c.post(_edit_url(tid, ids[1]),
               json={"content": "better answer", "access_code": CODE})
    assert r.status_code == 200
    assert r.json() == {"edited": True, "message": {
        "id": ids[1], "role": "assistant", "content": "better answer"}}
    assert [(m["role"], m["content"]) for m in memory.get_history(tid)] == [
        ("user", "first question"), ("assistant", "better answer")]


def test_message_edit_resend_reruns_from_edited_prompt(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    memory.add_message(tid, "user", "follow-up")
    memory.add_message(tid, "assistant", "follow-up answer")
    ids = [m["id"] for m in memory.get_history_with_ids(tid)]
    seen = _fake_turn(monkeypatch, reply="fixed answer")
    r = c.post(_edit_url(tid, ids[0]), json={
        "content": "first question, fixed", "resend": True,
        "model": "kilo:kilo-auto/free", "access_code": CODE})
    assert r.status_code == 200
    assert "fixed answer" in r.text
    assert '"done"' in r.text
    # The agent was fed history ending at the edited prompt: the stale
    # reply and the whole follow-up turn are gone, not fed back in.
    assert seen["history"] == [
        {"role": "user", "content": "first question, fixed"}]
    assert [(m["role"], m["content"]) for m in memory.get_history(tid)] == [
        ("user", "first question, fixed"), ("assistant", "fixed answer")]


def test_message_edit_resend_rejects_assistant_message(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    ids = [m["id"] for m in memory.get_history_with_ids(tid)]
    r = c.post(_edit_url(tid, ids[1]), json={
        "content": "rewritten", "resend": True, "access_code": CODE})
    assert r.status_code == 400
    # Nothing was edited on the rejected re-send.
    assert memory.get_history(tid)[1]["content"] == "first answer"


def test_message_edit_resend_blocked_prompt_returns_refusal(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    ids = [m["id"] for m in memory.get_history_with_ids(tid)]

    def _must_not_run(*args, **kwargs):
        raise AssertionError("agent must not run on a blocked prompt")

    from app import agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_turn", _must_not_run)
    r = c.post(_edit_url(tid, ids[0]), json={
        "content": "where can I buy a stolen credit card number?",
        "resend": True, "access_code": CODE})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is True
    # The edited prompt is kept, the old reply is gone, the refusal
    # is stored as the new reply.
    assert [(m["role"], m["content"]) for m in memory.get_history(tid)] == [
        ("user", "where can I buy a stolen credit card number?"),
        ("assistant", body["reply"])]


def test_message_edit_first_message_follows_auto_title(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    # _thread_client titles the thread "reopen me", not from the first
    # message — rename it to the auto title first so the follow works.
    memory.rename_thread(tid, "first question")
    ids = [m["id"] for m in memory.get_history_with_ids(tid)]
    r = c.post(_edit_url(tid, ids[0]), json={
        "content": "first question, fixed", "access_code": CODE})
    assert r.status_code == 200
    assert memory.get_thread(tid)["title"] == "first question, fixed"


def test_thread_patch_instructions(monkeypatch, tmp_path):
    c, tid = _thread_client(monkeypatch, tmp_path)
    r = c.patch(f"/api/threads/{tid}",
                json={"instructions": "Answer in French.",
                      "access_code": CODE})
    assert r.status_code == 200
    assert r.json()["thread"]["instructions"] == "Answer in French."
    # Title untouched, and the sidebar list carries instructions too.
    assert r.json()["thread"]["title"] == "reopen me"
    r = c.get(f"/api/threads?access_code={CODE}")
    assert r.json()["threads"][0]["instructions"] == "Answer in French."
    # An explicit empty value clears them; omitting both fields is a 400.
    r = c.patch(f"/api/threads/{tid}",
                json={"instructions": "", "access_code": CODE})
    assert r.status_code == 200
    assert r.json()["thread"]["instructions"] == ""
    r = c.patch(f"/api/threads/{tid}", json={"access_code": CODE})
    assert r.status_code == 400
    r = c.patch(f"/api/threads/{tid}",
                json={"title": "", "access_code": CODE})
    assert r.status_code == 400
    r = c.patch(f"/api/threads/{tid}",
                json={"instructions": "x", "access_code": "wrong"})
    assert r.status_code == 401
    r = c.patch("/api/threads/nope-not-real",
                json={"instructions": "x", "access_code": CODE})
    assert r.status_code == 404


def test_chat_passes_thread_instructions_to_agent(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    memory.set_thread_instructions(tid, "Answer in French.")
    seen = _fake_turn(monkeypatch, reply="bonjour")
    r = c.post("/api/chat", json={
        "message": "say hi", "thread_id": tid, "access_code": CODE})
    assert r.status_code == 200
    assert seen["extra_instructions"] == "Answer in French."
    # Regenerate re-runs with the same instructions.
    seen = _fake_turn(monkeypatch, reply="bonjour encore")
    r = c.post(f"/api/threads/{tid}/regenerate", json={"access_code": CODE})
    assert r.status_code == 200
    assert seen["extra_instructions"] == "Answer in French."


def test_run_turn_sends_thread_instructions_as_system_message(monkeypatch):
    from app import agent as agent_mod
    from app import models
    from tests.conftest import run
    seen = {}

    def _fake_chat(messages, max_tokens, temperature, tools,
                   progress=None, preferred_model="", on_token=None):
        seen["messages"] = list(messages)
        return "answer", "fake", [], None, False

    monkeypatch.setattr(models, "chat", _fake_chat)

    async def emit(ev):
        pass

    result = run(agent_mod.run_turn(
        [{"role": "user", "content": "hi"}], emit,
        extra_instructions="Answer in French."))
    assert result == "answer"
    systems = [m for m in seen["messages"] if m["role"] == "system"]
    assert len(systems) == 2
    assert "Answer in French." in systems[1]["content"]
    assert systems[0]["content"].startswith("You are an uncensored")
    # No instructions: one system message, the base prompt only.
    result = run(agent_mod.run_turn(
        [{"role": "user", "content": "hi"}], emit))
    assert result == "answer"
    assert len([m for m in seen["messages"]
                if m["role"] == "system"]) == 1


def test_import_bundle_restores_instructions_and_export_carries_them(monkeypatch, tmp_path):
    from app import memory
    c, tid = _thread_client(monkeypatch, tmp_path)
    memory.set_thread_instructions(tid, "Answer in French.")
    r = c.get(f"/api/threads/export-all?format=json&access_code={CODE}")
    assert r.status_code == 200
    bundle = r.json()
    assert bundle["threads"][0]["thread"]["instructions"] == "Answer in French."
    # Import the bundle into a fresh db: instructions survive the round trip.
    monkeypatch.setattr(memory, "DB", str(tmp_path / "fresh.db"))
    r = c.post("/api/threads/import",
               json={"threads": bundle["threads"], "access_code": CODE})
    assert r.status_code == 200
    new_id = r.json()["thread_ids"][0]
    assert memory.get_thread(new_id)["instructions"] == "Answer in French."
