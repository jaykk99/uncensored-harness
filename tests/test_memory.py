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
