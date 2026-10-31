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


def test_get_all_messages_empty_and_unknown(memdb):
    tid = memdb.create_thread("t")
    assert memdb.get_all_messages(tid) == []
    assert memdb.get_all_messages("nope-not-real") == []
