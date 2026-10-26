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
    assert set(row) == {"id", "title", "created"}
    assert row["title"] == "shape"
    assert isinstance(row["created"], float)


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
    assert set(hits[0]) == {"id", "title", "created", "snippet"}
    limited = memdb.search_threads("commonword", limit=3)
    assert len(limited) == 3
    assert [h["id"] for h in limited] == [h["id"] for h in hits[:3]]
