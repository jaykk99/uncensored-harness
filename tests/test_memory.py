"""Tests for app/memory.py — sqlite thread storage.

The DB path is module-level, so every test monkeypatches app.memory.DB to
a tmp file. Tests cover: thread creation, existence checks, message
ordering/limits, thread listing order/limits, and unicode content.
"""

from __future__ import annotations

import time

import pytest

from app import memory
from tests.conftest import run


@pytest.fixture
def memdb(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB", str(tmp_path / "t.db"))
    return memory


def test_create_thread_returns_id_and_exists(memdb):
    tid = memdb.create_thread("hello world")
    assert isinstance(tid, str) and len(tid) == 12
    assert memdb.thread_exists(tid) is True
    assert memdb.thread_exists("nope-not-real") is False


def test_create_thread_truncates_long_title(memdb):
    tid = memdb.create_thread("x" * 500)
    threads = memdb.list_threads()
    assert threads[0]["title"] == "x" * 60


def test_add_and_get_history_order(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "first")
    memdb.add_message(tid, "assistant", "second")
    memdb.add_message(tid, "user", "third")
    hist = memdb.get_history(tid)
    assert [(m["role"], m["content"]) for m in hist] == [
        ("user", "first"), ("assistant", "second"), ("user", "third")]


def test_history_limit_returns_most_recent(memdb):
    tid = memdb.create_thread("t")
    for i in range(10):
        memdb.add_message(tid, "user", f"msg-{i}")
    hist = memdb.get_history(tid, limit=4)
    assert [m["content"] for m in hist] == ["msg-6", "msg-7", "msg-8", "msg-9"]


def test_history_empty_thread(memdb):
    tid = memdb.create_thread("t")
    assert memdb.get_history(tid) == []


def test_list_threads_newest_first_and_capped(memdb):
    ids = [memdb.create_thread(f"t-{i}") for i in range(55)]
    threads = memdb.list_threads()
    assert len(threads) == 50  # capped
    # Newest first: the last created id should be at the top.
    assert threads[0]["id"] == ids[-1]
    assert threads[-1]["id"] == ids[5]


def test_list_threads_shape(memdb):
    tid = memdb.create_thread("shape")
    (row,) = [t for t in memdb.list_threads() if t["id"] == tid]
    assert set(row) == {"id", "title", "created", "pinned", "archived"}
    assert row["title"] == "shape"
    assert row["pinned"] is False
    assert row["archived"] is False
    assert isinstance(row["created"], float)


def test_pin_sorts_above_newer_unpinned(memdb):
    old = memdb.create_thread("old but important")
    new = memdb.create_thread("newer")
    assert [t["id"] for t in memdb.list_threads()] == [new, old]
    assert memdb.set_thread_pinned(old, True) is True
    assert memdb.get_thread(old)["pinned"] is True
    assert [t["id"] for t in memdb.list_threads()] == [old, new]
    assert memdb.set_thread_pinned(old, False) is True
    assert [t["id"] for t in memdb.list_threads()] == [new, old]


def test_pin_unknown_thread_is_false(memdb):
    assert memdb.set_thread_pinned("nope-not-real", True) is False


def test_archive_hides_from_list_and_archive_view_lists_it(memdb):
    old = memdb.create_thread("finished chat")
    new = memdb.create_thread("active chat")
    assert memdb.set_thread_archived(old, True) is True
    assert memdb.get_thread(old)["archived"] is True
    # Sidebar list drops it; the archive view is the only place it shows.
    assert [t["id"] for t in memdb.list_threads()] == [new]
    assert [t["id"] for t in memdb.list_archived_threads()] == [old]
    # Unarchiving brings it straight back to the sidebar list.
    assert memdb.set_thread_archived(old, False) is True
    assert [t["id"] for t in memdb.list_threads()] == [new, old]
    assert memdb.list_archived_threads() == []


def test_archive_unknown_thread_is_false(memdb):
    assert memdb.set_thread_archived("nope-not-real", True) is False


def test_duplicate_of_archived_thread_starts_unarchived(memdb):
    tid = memdb.create_thread("archived source")
    memdb.set_thread_archived(tid, True)
    new_id = memdb.duplicate_thread(tid)
    assert memdb.get_thread(new_id)["archived"] is False
    assert memdb.get_thread(tid)["archived"] is True
    assert new_id in [t["id"] for t in memdb.list_threads()]


def test_search_still_finds_archived_thread(memdb):
    # Archiving must never strand a thread: search is the recovery path
    # for a chat the user archived and then forgot the title of.
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "the quota script broke again")
    memdb.set_thread_archived(tid, True)
    hits = memdb.search_threads("quota")
    assert [h["id"] for h in hits] == [tid]
    assert hits[0]["archived"] is True


def test_old_db_without_archived_column_migrates(memdb, tmp_path, monkeypatch):
    # DBs created before archiving have neither new column at all.
    import sqlite3
    path = str(tmp_path / "old2.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT, created REAL)")
    c.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
              " thread_id TEXT, role TEXT, content TEXT, created REAL)")
    c.execute("INSERT INTO threads VALUES (?,?,?)", ("oldthreadid2", "legacy", 1.0))
    c.commit()
    c.close()
    monkeypatch.setattr(memdb, "DB", path)
    assert memdb.get_thread("oldthreadid2")["archived"] is False
    assert memdb.set_thread_archived("oldthreadid2", True) is True
    assert memdb.list_threads() == []
    assert [t["id"] for t in memdb.list_archived_threads()] == ["oldthreadid2"]


def test_duplicate_of_pinned_thread_starts_unpinned(memdb):
    tid = memdb.create_thread("pinned source")
    memdb.set_thread_pinned(tid, True)
    new_id = memdb.duplicate_thread(tid)
    assert memdb.get_thread(new_id)["pinned"] is False
    assert memdb.get_thread(tid)["pinned"] is True


def test_old_db_without_pinned_column_migrates(memdb, tmp_path, monkeypatch):
    # DBs created before pinning have no pinned column at all.
    import sqlite3
    path = str(tmp_path / "old.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT, created REAL)")
    c.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
              " thread_id TEXT, role TEXT, content TEXT, created REAL)")
    c.execute("INSERT INTO threads VALUES (?,?,?)", ("oldthreadid1", "legacy", 1.0))
    c.commit()
    c.close()
    monkeypatch.setattr(memdb, "DB", path)
    assert memdb.get_thread("oldthreadid1")["pinned"] is False
    assert memdb.set_thread_pinned("oldthreadid1", True) is True
    assert memdb.list_threads()[0]["pinned"] is True


def test_unicode_and_newlines_roundtrip(memdb):
    tid = memdb.create_thread("t")
    content = "héllo wörld\nline2\t\"quotes\" 'apos' 😀"
    memdb.add_message(tid, "user", content)
    assert memdb.get_history(tid)[0]["content"] == content


def test_threads_are_isolated(memdb):
    a = memdb.create_thread("a")
    b = memdb.create_thread("b")
    memdb.add_message(a, "user", "only-a")
    assert memdb.get_history(b) == []
    assert memdb.get_history(a)[0]["content"] == "only-a"


def test_history_default_limit(memdb):
    tid = memdb.create_thread("t")
    for i in range(70):
        memdb.add_message(tid, "user", f"m{i}")
    assert len(memdb.get_history(tid)) == 60


def test_history_offset_pages_backwards(memdb):
    tid = memdb.create_thread("t")
    for i in range(10):
        memdb.add_message(tid, "user", f"m{i}")
    # Newest page, then the older page: together they cover the thread
    # exactly once, in order, with no overlap or gap.
    newest = [m["content"] for m in memdb.get_history(tid, limit=4)]
    older = [m["content"] for m in memdb.get_history(tid, limit=4, offset=4)]
    oldest = [m["content"] for m in memdb.get_history(tid, limit=4, offset=8)]
    assert newest == ["m6", "m7", "m8", "m9"]
    assert older == ["m2", "m3", "m4", "m5"]
    assert oldest == ["m0", "m1"]
    assert memdb.get_history(tid, limit=4, offset=10) == []
    # Negative offsets clamp to 0: same as the newest page.
    assert [m["content"] for m in memdb.get_history(tid, limit=4, offset=-5)] == newest


def test_count_messages(memdb):
    tid = memdb.create_thread("t")
    assert memdb.count_messages(tid) == 0
    memdb.add_message(tid, "user", "one")
    memdb.add_message(tid, "assistant", "two")
    assert memdb.count_messages(tid) == 2
    assert memdb.count_messages("nope-not-real") == 0


def test_get_thread_and_unknown(memdb):
    tid = memdb.create_thread("named")
    got = memdb.get_thread(tid)
    assert got["id"] == tid and got["title"] == "named"
    assert isinstance(got["created"], float)
    assert memdb.get_thread("nope-not-real") is None


def test_delete_thread_removes_messages(memdb):
    tid = memdb.create_thread("gone")
    memdb.add_message(tid, "user", "hello")
    assert memdb.delete_thread(tid) is True
    assert memdb.thread_exists(tid) is False
    assert memdb.get_history(tid) == []
    assert memdb.get_thread(tid) is None
    assert memdb.delete_thread(tid) is False


def test_delete_thread_leaves_other_threads(memdb):
    a = memdb.create_thread("a")
    b = memdb.create_thread("b")
    memdb.add_message(b, "user", "keep me")
    assert memdb.delete_thread(a) is True
    assert memdb.get_history(b)[0]["content"] == "keep me"


def test_rename_thread_truncates_and_unknown(memdb):
    tid = memdb.create_thread("old")
    assert memdb.rename_thread(tid, "x" * 500) is True
    assert memdb.get_thread(tid)["title"] == "x" * 60
    assert memdb.rename_thread("nope-not-real", "new") is False


def test_search_matches_title(memdb):
    tid = memdb.create_thread("deploy railway notes")
    memdb.add_message(tid, "user", "unrelated chatter")
    hits = memdb.search_threads("railway")
    assert [h["id"] for h in hits] == [tid]
    assert "railway" in hits[0]["snippet"].lower()


def test_search_matches_message_content_case_insensitive(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "the QUOTA script broke again")
    other = memdb.create_thread("u")
    memdb.add_message(other, "user", "nothing relevant")
    hits = memdb.search_threads("quota")
    assert [h["id"] for h in hits] == [tid]
    assert "quota" in hits[0]["snippet"].lower()


def test_search_dedupes_thread_with_many_matching_messages(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "banana one")
    memdb.add_message(tid, "user", "banana two")
    assert len(memdb.search_threads("banana")) == 1


def test_search_empty_and_no_match(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "hello")
    assert memdb.search_threads("") == []
    assert memdb.search_threads("   ") == []
    assert memdb.search_threads("zzz-no-match") == []


def test_search_escapes_like_wildcards(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "100% sure")
    memdb.add_message(tid, "user", "plain text")
    # A literal "%" must not match every message in the thread.
    hits = memdb.search_threads("%")
    assert len(hits) == 1 and hits[0]["id"] == tid
    assert "%" in hits[0]["snippet"]
    assert memdb.search_threads("_") == []


def test_search_limit_and_shape(memdb):
    ids = []
    for i in range(25):
        tid = memdb.create_thread(f"t-{i}")
        memdb.add_message(tid, "user", "commonword here")
        ids.append(tid)
    hits = memdb.search_threads("commonword")
    assert len(hits) == 20  # default limit
    assert set(hits[0]) == {"id", "title", "created", "archived", "pinned",
                            "snippet"}
    limited = memdb.search_threads("commonword", limit=3)
    assert len(limited) == 3
    assert [h["id"] for h in limited] == [h["id"] for h in hits[:3]]


def test_search_sorts_pinned_above_newer_unpinned(memdb):
    # The sidebar lists pinned threads first, but search used to sort
    # purely by newest matching message: a pinned older chat sank below
    # a newer unpinned one, and its hit carried no pinned flag at all,
    # so the sidebar star rendered unpinned for a pinned thread.
    old = memdb.create_thread("old pinned")
    memdb.add_message(old, "user", "commonword old")
    time.sleep(0.01)
    new = memdb.create_thread("new unpinned")
    memdb.add_message(new, "user", "commonword new")
    assert memdb.set_thread_pinned(old, True) is True
    hits = memdb.search_threads("commonword")
    assert [h["id"] for h in hits] == [old, new]
    assert hits[0]["pinned"] is True
    assert hits[1]["pinned"] is False


def test_search_finds_message_less_thread_by_title(memdb):
    # Search joined threads to messages, so a thread with no messages
    # yet was invisible even when its title matched exactly.
    tid = memdb.create_thread("railway deploy notes")
    other = memdb.create_thread("other")
    memdb.add_message(other, "user", "unrelated")
    hits = memdb.search_threads("railway")
    assert [h["id"] for h in hits] == [tid]
    assert "railway" in hits[0]["snippet"].lower()
    assert hits[0]["pinned"] is False


def test_search_thread_messages_returns_ids_positions_snippets(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "first about bananas")
    memdb.add_message(tid, "assistant", "unrelated reply")
    memdb.add_message(tid, "user", "second BANANAS note")
    other = memdb.create_thread("u")
    memdb.add_message(other, "user", "bananas elsewhere")
    res = memdb.search_thread_messages(tid, "bananas")
    # Only this thread's hits, oldest-first, with thread positions
    # (1 and 3 — the unrelated middle message still counts).
    assert res["total"] == 2
    assert res["thread_total"] == 3
    assert [(m["position"], m["role"]) for m in res["matches"]] == [
        (1, "user"), (3, "user")]
    assert all("bananas" in m["snippet"].lower() for m in res["matches"])
    assert all("id" in m for m in res["matches"])


def test_search_thread_messages_empty_wildcards_and_limit(memdb):
    tid = memdb.create_thread("t")
    for i in range(5):
        memdb.add_message(tid, "user", f"commonword {i}")
    memdb.add_message(tid, "user", "100% sure")
    assert memdb.search_thread_messages(tid, "")["matches"] == []
    assert memdb.search_thread_messages(tid, "   ")["total"] == 0
    assert memdb.search_thread_messages(tid, "zzz-no-match")["total"] == 0
    # A literal "%" matches only the message that contains one.
    pct = memdb.search_thread_messages(tid, "%")
    assert pct["total"] == 1 and "%" in pct["matches"][0]["snippet"]
    assert memdb.search_thread_messages(tid, "_")["total"] == 0
    # total counts every match even when limit caps the returned list.
    capped = memdb.search_thread_messages(tid, "commonword", limit=2)
    assert capped["total"] == 5 and len(capped["matches"]) == 2


def test_search_thread_messages_snippet_windows_around_match(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "x" * 100 + "needle" + "y" * 100)
    hit = memdb.search_thread_messages(tid, "needle")["matches"][0]
    assert hit["snippet"].startswith("...")
    assert "needle" in hit["snippet"]
    assert len(hit["snippet"]) < 200  # windowed, not the whole message


def test_search_all_messages_returns_per_message_hits_newest_first(memdb):
    a = memdb.create_thread("first chat")
    memdb.add_message(a, "user", "deploy the widget")
    memdb.add_message(a, "assistant", "widget deployed")
    memdb.add_message(a, "user", "widget broke again")
    b = memdb.create_thread("second chat")
    memdb.add_message(b, "assistant", "the widget needs a rebuild")
    res = memdb.search_all_messages("widget")
    assert res["total"] == 4
    # Newest match first: the rebuild answer, then the broke note,
    # then the deployed reply, then the first question.
    assert [(m["thread_id"], m["position"], m["role"])
            for m in res["matches"]] == [
        (b, 1, "assistant"), (a, 3, "user"),
        (a, 2, "assistant"), (a, 1, "user")]
    assert [m["thread_total"] for m in res["matches"]] == [1, 3, 3, 3]
    assert res["matches"][0]["thread_title"] == "second chat"
    assert all("widget" in m["snippet"].lower() for m in res["matches"])
    assert all("id" in m for m in res["matches"])


def test_search_all_messages_empty_wildcards_and_limit(memdb):
    tid = memdb.create_thread("t")
    for i in range(5):
        memdb.add_message(tid, "user", f"commonword {i}")
    memdb.add_message(tid, "user", "100% sure")
    assert memdb.search_all_messages("")["matches"] == []
    assert memdb.search_all_messages("   ")["total"] == 0
    assert memdb.search_all_messages("zzz-no-match")["total"] == 0
    # A literal "%" matches only the message that contains one.
    pct = memdb.search_all_messages("%")
    assert pct["total"] == 1 and "%" in pct["matches"][0]["snippet"]
    assert memdb.search_all_messages("_")["total"] == 0
    # total counts every match even when limit caps the returned list.
    capped = memdb.search_all_messages("commonword", limit=2)
    assert capped["total"] == 5 and len(capped["matches"]) == 2


def test_search_thread_messages_pages_with_after_id(memdb):
    tid = memdb.create_thread("t")
    for i in range(5):
        memdb.add_message(tid, "user", f"commonword {i}")
    memdb.add_message(tid, "user", "unrelated")
    # Page 1: the two oldest matches, with a cursor for page 2.
    p1 = memdb.search_thread_messages(tid, "commonword", limit=2)
    assert p1["total"] == 5 and len(p1["matches"]) == 2
    assert p1["has_more"] is True
    assert p1["next_after_id"] == p1["matches"][-1]["id"]
    assert [m["position"] for m in p1["matches"]] == [1, 2]
    # Page 2: the next two, oldest-first, no repeats.
    p2 = memdb.search_thread_messages(tid, "commonword", limit=2,
                                      after_id=p1["next_after_id"])
    assert p2["total"] == 5 and len(p2["matches"]) == 2
    assert p2["has_more"] is True
    assert [m["position"] for m in p2["matches"]] == [3, 4]
    # Page 3: the last match; the cursor is spent.
    p3 = memdb.search_thread_messages(tid, "commonword", limit=2,
                                      after_id=p2["next_after_id"])
    assert len(p3["matches"]) == 1
    assert [m["position"] for m in p3["matches"]] == [5]
    assert p3["has_more"] is False and p3["next_after_id"] is None
    # Chained pages cover every match exactly once.
    seen = [m["id"] for p in (p1, p2, p3) for m in p["matches"]]
    assert len(seen) == len(set(seen)) == 5
    # An empty query still reports no next page.
    assert memdb.search_thread_messages(
        tid, "", after_id=1)["has_more"] is False


def test_search_all_messages_pages_with_before_id(memdb):
    a = memdb.create_thread("first chat")
    for i in range(3):
        memdb.add_message(a, "user", f"commonword {i}")
    b = memdb.create_thread("second chat")
    for i in range(2):
        memdb.add_message(b, "assistant", f"commonword {i}")
    # 5 matches, newest first: the two from b, then the three from a.
    p1 = memdb.search_all_messages("commonword", limit=2)
    assert p1["total"] == 5 and len(p1["matches"]) == 2
    assert p1["has_more"] is True
    assert p1["next_before_id"] == p1["matches"][-1]["id"]
    assert [m["thread_id"] for m in p1["matches"]] == [b, b]
    p2 = memdb.search_all_messages("commonword", limit=2,
                                   before_id=p1["next_before_id"])
    assert p2["total"] == 5 and len(p2["matches"]) == 2
    assert p2["has_more"] is True
    assert [m["thread_id"] for m in p2["matches"]] == [a, a]
    p3 = memdb.search_all_messages("commonword", limit=2,
                                   before_id=p2["next_before_id"])
    assert len(p3["matches"]) == 1 and p3["has_more"] is False
    assert p3["next_before_id"] is None
    # Chained pages cover every match exactly once, newest first.
    # Every message here was stored within one second, so the
    # (created, id) cursor did the work, not the timestamp alone.
    seen = [m["id"] for p in (p1, p2, p3) for m in p["matches"]]
    assert len(seen) == len(set(seen)) == 5
    assert seen == sorted(seen, reverse=True)


def test_search_all_messages_unknown_before_id_is_empty_page(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "commonword here")
    res = memdb.search_all_messages("commonword", before_id=999999)
    assert res["matches"] == [] and res["total"] == 1
    assert res["has_more"] is False and res["next_before_id"] is None
    # One page that fits reports no next page either.
    single = memdb.search_all_messages("commonword")
    assert len(single["matches"]) == 1 and single["has_more"] is False
    assert single["next_before_id"] is None


def test_search_all_messages_includes_archived_with_flag(memdb):
    tid = memdb.create_thread("old chat")
    memdb.add_message(tid, "user", "archive me needle")
    memdb.set_thread_archived(tid, True)
    other = memdb.create_thread("live chat")
    memdb.add_message(other, "user", "live needle here")
    hits = memdb.search_all_messages("needle")["matches"]
    assert len(hits) == 2
    by_thread = {h["thread_id"]: h for h in hits}
    assert by_thread[tid]["archived"] is True
    assert by_thread[other]["archived"] is False


def test_search_all_messages_snippet_windows_around_match(memdb):
    tid = memdb.create_thread("t")
    memdb.add_message(tid, "user", "x" * 100 + "needle" + "y" * 100)
    hit = memdb.search_all_messages("needle")["matches"][0]
    assert hit["snippet"].startswith("...")
    assert "needle" in hit["snippet"]
    assert len(hit["snippet"]) < 200  # windowed, not the whole message


def test_get_all_messages_returns_everything_in_order(memdb):
    tid = memdb.create_thread("t")
    for i in range(5):
        memdb.add_message(tid, "user" if i % 2 == 0 else "assistant",
                          f"msg-{i}")
    assert [(m["role"], m["content"]) for m in memdb.get_all_messages(tid)] == [
        ("user", "msg-0"), ("assistant", "msg-1"), ("user", "msg-2"),
        ("assistant", "msg-3"), ("user", "msg-4")]
    # No page cap, unlike get_history's 200: a long thread exports whole.
    assert memdb.get_all_messages(tid) != memdb.get_history(tid, limit=2)


def test_duplicate_copies_messages_in_order_and_leaves_original(memdb):
    tid = memdb.create_thread("branch me")
    memdb.add_message(tid, "user", "first")
    memdb.add_message(tid, "assistant", "second")
    new_id = memdb.duplicate_thread(tid)
    assert new_id is not None and new_id != tid
    assert memdb.get_thread(new_id)["title"] == "branch me (copy)"
    assert [(m["role"], m["content"]) for m in memdb.get_history(new_id)] == [
        ("user", "first"), ("assistant", "second")]
    # Original untouched, and the copy diverges independently.
    assert memdb.get_history(tid)[0]["content"] == "first"
    memdb.add_message(new_id, "user", "only in copy")
    assert len(memdb.get_history(tid)) == 2
    assert len(memdb.get_history(new_id)) == 3


def test_duplicate_empty_thread_and_unknown(memdb):
    tid = memdb.create_thread("empty")
    new_id = memdb.duplicate_thread(tid)
    assert memdb.get_thread(new_id)["title"] == "empty (copy)"
    assert memdb.get_history(new_id) == []
    assert memdb.duplicate_thread("nope-not-real") is None


def test_duplicate_title_suffix_survives_cap(memdb):
    tid = memdb.create_thread("x" * 60)
    new_id = memdb.duplicate_thread(tid)
    title = memdb.get_thread(new_id)["title"]
    assert title == "x" * 53 + " (copy)"
    assert len(title) == 60


def test_import_thread_restores_messages_in_order_under_new_id(memdb):
    msgs = [{"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"}]
    new_id = memdb.import_thread("restored chat", msgs)
    assert isinstance(new_id, str) and len(new_id) == 12
    thread = memdb.get_thread(new_id)
    assert thread["title"] == "restored chat"
    assert thread["pinned"] is False and thread["archived"] is False
    assert [(m["role"], m["content"]) for m in memdb.get_history(new_id)] == [
        ("user", "hello"), ("assistant", "hi back")]


def test_import_thread_empty_title_and_long_title(memdb):
    new_id = memdb.import_thread("", [])
    assert memdb.get_thread(new_id)["title"] == "untitled"
    assert memdb.get_history(new_id) == []
    long_id = memdb.import_thread("y" * 100, [])
    assert memdb.get_thread(long_id)["title"] == "y" * 60


def test_get_all_messages_empty_and_unknown(memdb):
    tid = memdb.create_thread("t")
    assert memdb.get_all_messages(tid) == []
    assert memdb.get_all_messages("nope-not-real") == []


def _prune_thread(memdb):
    tid = memdb.create_thread("prune me")
    for role, content in [("user", "q1"), ("assistant", "a1"),
                          ("user", "q2"), ("assistant", "a2")]:
        memdb.add_message(tid, role, content)
    ids = [m["id"] for m in memdb.get_history_with_ids(tid)]
    assert [m["content"] for m in memdb.get_history_with_ids(tid)] == [
        "q1", "a1", "q2", "a2"]
    # get_history itself stays id-free: it feeds the model, which needs
    # strict OpenAI message format.
    assert memdb.get_history(tid) == [
        {"role": r, "content": c} for r, c in
        [("user", "q1"), ("assistant", "a1"),
         ("user", "q2"), ("assistant", "a2")]]
    return tid, ids


def test_history_with_ids_pages_like_history(memdb):
    tid, ids = _prune_thread(memdb)
    page = memdb.get_history_with_ids(tid, limit=2)
    assert [m["id"] for m in page] == ids[2:]
    assert [m["content"] for m in page] == ["q2", "a2"]
    older = memdb.get_history_with_ids(tid, limit=2, offset=2)
    assert [m["content"] for m in older] == ["q1", "a1"]


def test_delete_message_removes_only_that_row(memdb):
    tid, ids = _prune_thread(memdb)
    assert memdb.delete_message(tid, ids[1]) is True
    assert [m["content"] for m in memdb.get_history(tid)] == [
        "q1", "q2", "a2"]
    # Survivors keep their ids: a second delete still names its row.
    assert memdb.delete_message(tid, ids[3]) is True
    assert [m["content"] for m in memdb.get_history(tid)] == ["q1", "q2"]


def test_delete_message_unknown_or_foreign_is_false(memdb):
    tid, ids = _prune_thread(memdb)
    other = memdb.create_thread("other")
    assert memdb.delete_message(tid, 999999) is False
    # An id from another thread must not delete anything here.
    assert memdb.delete_message(other, ids[0]) is False
    assert len(memdb.get_history(tid)) == 4


def test_truncate_thread_from_cuts_anchor_and_tail(memdb):
    tid, ids = _prune_thread(memdb)
    assert memdb.truncate_thread_from(tid, ids[2]) == 2
    assert [m["content"] for m in memdb.get_history(tid)] == ["q1", "a1"]


def test_truncate_unknown_or_foreign_cuts_nothing(memdb):
    tid, ids = _prune_thread(memdb)
    other = memdb.create_thread("other")
    assert memdb.truncate_thread_from(tid, 999999) == 0
    assert memdb.truncate_thread_from(other, ids[1]) == 0
    assert len(memdb.get_history(tid)) == 4
    assert memdb.get_history(other) == []


def test_history_with_ids_before_id_walks_to_first_message(memdb):
    tid, ids = _prune_thread(memdb)
    # Newest page, then cursor pages walk back disjointly to the start.
    page = memdb.get_history_with_ids(tid, limit=2)
    assert [m["id"] for m in page] == ids[2:]
    older = memdb.get_history_with_ids(tid, limit=2, before_id=page[0]["id"])
    assert [m["id"] for m in older] == ids[:2]
    assert [m["content"] for m in older] == ["q1", "a1"]
    assert memdb.get_history_with_ids(
        tid, limit=2, before_id=older[0]["id"]) == []


def test_message_position_counts_from_oldest(memdb):
    tid, ids = _prune_thread(memdb)
    assert memdb.message_position(tid, ids[0]) == {"position": 1, "total": 4}
    assert memdb.message_position(tid, ids[2]) == {"position": 3, "total": 4}
    assert memdb.message_position(tid, ids[3]) == {"position": 4, "total": 4}
    # Unknown id, foreign id, and unknown thread all resolve to None:
    # a position must never be guessed for a row that is not in this
    # thread, or a jump would land on the wrong turn.
    assert memdb.message_position(tid, 999999) is None
    other = memdb.create_thread("other")
    assert memdb.message_position(other, ids[0]) is None
    assert memdb.message_position("nope-not-real", ids[0]) is None


def test_message_position_shifts_after_delete(memdb):
    tid, ids = _prune_thread(memdb)
    memdb.delete_message(tid, ids[1])
    assert memdb.message_position(tid, ids[0]) == {"position": 1, "total": 3}
    assert memdb.message_position(tid, ids[2]) == {"position": 2, "total": 3}
    assert memdb.message_position(tid, ids[1]) is None


def test_history_with_ids_before_id_wins_over_offset(memdb):
    tid, ids = _prune_thread(memdb)
    # offset=0 alone would return the newest page; the cursor must win.
    page = memdb.get_history_with_ids(tid, limit=2, offset=0,
                                      before_id=ids[2])
    assert [m["content"] for m in page] == ["q1", "a1"]
    # Ids are a global sequence, so a cursor from another thread is
    # still just an id comparison — but rows stay scoped to this
    # thread: the other thread's message must never leak in.
    other = memdb.create_thread("other")
    memdb.add_message(other, "user", "elsewhere")
    other_id = memdb.get_history_with_ids(other)[0]["id"]
    page = memdb.get_history_with_ids(tid, limit=2, before_id=other_id)
    assert [m["content"] for m in page] == ["q2", "a2"]
    assert all(m["content"] != "elsewhere" for m in page)


def test_last_message_returns_newest_with_id(memdb):
    tid = memdb.create_thread("t")
    assert memdb.last_message(tid) is None
    memdb.add_message(tid, "user", "q")
    memdb.add_message(tid, "assistant", "a")
    last = memdb.last_message(tid)
    assert last["role"] == "assistant" and last["content"] == "a"
    assert isinstance(last["id"], int)
    assert memdb.last_message("nope-not-real") is None


def test_drop_last_assistant_only_drops_assistant_tail(memdb):
    tid = memdb.create_thread("t")
    assert memdb.drop_last_assistant(tid) is False  # empty thread
    memdb.add_message(tid, "user", "q1")
    # Tail is the user's prompt: dropping must not eat it — the reply
    # never landed, so there is nothing stored to replace.
    assert memdb.drop_last_assistant(tid) is False
    assert memdb.get_history(tid) == [{"role": "user", "content": "q1"}]
    memdb.add_message(tid, "assistant", "a1")
    memdb.add_message(tid, "user", "q2")
    # Newest is a user message again: the older assistant reply is
    # not the tail, so it stays.
    assert memdb.drop_last_assistant(tid) is False
    memdb.add_message(tid, "assistant", "a2")
    assert memdb.drop_last_assistant(tid) is True
    assert [(m["role"], m["content"]) for m in memdb.get_history(tid)] == [
        ("user", "q1"), ("assistant", "a1"), ("user", "q2")]
    # The other thread's tail is untouched.
    other = memdb.create_thread("other")
    memdb.add_message(other, "assistant", "elsewhere")
    assert memdb.drop_last_assistant(tid) is False
    assert memdb.get_history(other) == [
        {"role": "assistant", "content": "elsewhere"}]


def test_get_message_returns_row_or_none(memdb):
    tid, ids = _prune_thread(memdb)
    other = memdb.create_thread("other")
    assert memdb.get_message(tid, ids[1]) == {
        "id": ids[1], "role": "assistant", "content": "a1"}
    assert memdb.get_message(tid, 999999) is None
    # A foreign id never resolves here.
    assert memdb.get_message(other, ids[0]) is None


def test_edit_message_replaces_content_and_reports_old(memdb):
    tid, ids = _prune_thread(memdb)
    edited = memdb.edit_message(tid, ids[2], "q2 fixed")
    assert edited == {"id": ids[2], "role": "user", "content": "q2 fixed",
                      "old_content": "q2"}
    assert [m["content"] for m in memdb.get_history(tid)] == [
        "q1", "a1", "q2 fixed", "a2"]


def test_edit_message_unknown_or_foreign_edits_nothing(memdb):
    tid, ids = _prune_thread(memdb)
    other = memdb.create_thread("other")
    assert memdb.edit_message(tid, 999999, "nope") is None
    assert memdb.edit_message(other, ids[0], "nope") is None
    assert [m["content"] for m in memdb.get_history(tid)] == [
        "q1", "a1", "q2", "a2"]


def test_edit_first_message_follows_auto_title(memdb):
    tid = memdb.create_thread("what is the capital of France?")
    memdb.add_message(tid, "user", "what is the capital of France?")
    memdb.add_message(tid, "assistant", "Paris.")
    ids = [m["id"] for m in memdb.get_history_with_ids(tid)]
    memdb.edit_message(tid, ids[0], "what is the capital of Germany?")
    assert memdb.get_thread(tid)["title"] == "what is the capital of Germany?"
    # Editing a later message leaves the title alone.
    memdb.edit_message(tid, ids[1], "Berlin, actually.")
    assert memdb.get_thread(tid)["title"] == "what is the capital of Germany?"


def test_edit_first_message_leaves_renamed_title_alone(memdb):
    tid = memdb.create_thread("what is the capital of France?")
    memdb.add_message(tid, "user", "what is the capital of France?")
    memdb.add_message(tid, "assistant", "Paris.")
    memdb.rename_thread(tid, "capitals")
    ids = [m["id"] for m in memdb.get_history_with_ids(tid)]
    memdb.edit_message(tid, ids[0], "what is the capital of Germany?")
    assert memdb.get_thread(tid)["title"] == "capitals"


def test_delete_after_keeps_anchor_and_cuts_tail(memdb):
    tid, ids = _prune_thread(memdb)
    assert memdb.delete_after(tid, ids[1]) == 2
    assert [m["content"] for m in memdb.get_history(tid)] == ["q1", "a1"]
    # Cutting after the last message deletes nothing.
    assert memdb.delete_after(tid, ids[1]) == 0


def test_delete_after_unknown_or_foreign_cuts_nothing(memdb):
    tid, ids = _prune_thread(memdb)
    other = memdb.create_thread("other")
    assert memdb.delete_after(tid, 999999) == 0
    assert memdb.delete_after(other, ids[0]) == 0
    assert len(memdb.get_history(tid)) == 4
