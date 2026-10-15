"""Thread storage: tiny sqlite db with messages per thread."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid

DB = os.path.join(os.path.dirname(__file__), "..", "harness.db")


def _conn():
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS threads
                 (id TEXT PRIMARY KEY, title TEXT, created REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT,
                  role TEXT, content TEXT, created REAL)""")
    return c


def create_thread(title: str) -> str:
    tid = uuid.uuid4().hex[:12]
    c = _conn()
    c.execute("INSERT INTO threads VALUES (?,?,?)", (tid, title[:60], time.time()))
    c.commit()
    c.close()
    return tid


def thread_exists(tid: str) -> bool:
    c = _conn()
    row = c.execute("SELECT 1 FROM threads WHERE id=?", (tid,)).fetchone()
    c.close()
    return bool(row)


def add_message(tid: str, role: str, content: str) -> None:
    c = _conn()
    c.execute("INSERT INTO messages (thread_id, role, content, created) VALUES (?,?,?,?)",
              (tid, role, content, time.time()))
    c.commit()
    c.close()


def get_history(tid: str, limit: int = 60) -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?",
        (tid, limit)).fetchall()
    c.close()
    return [{"role": r, "content": co} for r, co in reversed(rows)]


def list_threads() -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT id, title, created FROM threads ORDER BY created DESC LIMIT 50").fetchall()
    c.close()
    return [{"id": r[0], "title": r[1], "created": r[2]} for r in rows]
