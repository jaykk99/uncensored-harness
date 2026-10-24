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


def get_thread(tid: str) -> dict | None:
    c = _conn()
    row = c.execute(
        "SELECT id, title, created FROM threads WHERE id=?", (tid,)).fetchone()
    c.close()
    if not row:
        return None
    return {"id": row[0], "title": row[1], "created": row[2]}


def delete_thread(tid: str) -> bool:
    """Delete a thread and all its messages. Returns False if unknown."""
    c = _conn()
    cur = c.execute("DELETE FROM threads WHERE id=?", (tid,))
    deleted = cur.rowcount > 0
    if deleted:
        c.execute("DELETE FROM messages WHERE thread_id=?", (tid,))
    c.commit()
    c.close()
    return deleted


def rename_thread(tid: str, title: str) -> bool:
    """Rename a thread (title capped at 60 like create_thread).

    Returns False if the thread does not exist.
    """
    c = _conn()
    cur = c.execute("UPDATE threads SET title=? WHERE id=?",
                    (title[:60], tid))
    renamed = cur.rowcount > 0
    c.commit()
    c.close()
    return renamed


def list_threads() -> list[dict]:
    c = _conn()
    rows = c.execute(
        "SELECT id, title, created FROM threads ORDER BY created DESC LIMIT 50").fetchall()
    c.close()
    return [{"id": r[0], "title": r[1], "created": r[2]} for r in rows]
