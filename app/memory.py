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


def get_history(tid: str, limit: int = 60, offset: int = 0) -> list[dict]:
    """Most recent messages, oldest-first within the page.

    Reopening a thread used to fetch only the newest 60 messages with no
    way back: anything older was silently unreachable in the UI. `offset`
    pages backwards through history (offset = how many of the newest
    messages to skip), so the UI can offer "load older messages" until
    the whole thread is on screen. Shape stays {role, content} only —
    this history is also fed to the model, which needs strict OpenAI
    message format, so no id or metadata keys may be added here.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    c = _conn()
    rows = c.execute(
        "SELECT role, content FROM messages WHERE thread_id=? "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (tid, limit, offset)).fetchall()
    c.close()
    return [{"role": r, "content": co} for r, co in reversed(rows)]


def get_all_messages(tid: str) -> list[dict]:
    """Every message in a thread, oldest-first, with no page cap.

    get_history() caps at 200 and pages newest-first, which is right
    for the model feed and the UI pager but wrong for an export: a
    saved transcript must carry the whole thread in reading order, so
    this query has no LIMIT and sorts ascending. Same {role, content}
    shape as get_history.
    """
    c = _conn()
    rows = c.execute(
        "SELECT role, content FROM messages WHERE thread_id=? "
        "ORDER BY id ASC",
        (tid,)).fetchall()
    c.close()
    return [{"role": r, "content": co} for r, co in rows]


def count_messages(tid: str) -> int:
    """Total messages in a thread, for paging decisions in the UI."""
    c = _conn()
    row = c.execute(
        "SELECT COUNT(*) FROM messages WHERE thread_id=?", (tid,)).fetchone()
    c.close()
    return int(row[0]) if row else 0


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


def search_threads(query: str, limit: int = 20) -> list[dict]:
    """Find threads by title or message content.

    The sidebar only lists the newest 50 threads, so anything older was
    unreachable. A hit returns the thread plus a short excerpt of a
    matching message, so the user can tell *why* it matched before
    opening it. LIKE wildcards in the query are escaped: a literal "%"
    in the search text must not match everything.
    """
    q = (query or "").strip()
    if not q:
        return []
    esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pat = f"%{esc}%"
    limit = max(1, min(limit, 50))
    c = _conn()
    rows = c.execute(
        """SELECT t.id, t.title, t.created, m.content
             FROM threads t JOIN messages m ON m.thread_id = t.id
            WHERE t.title LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\'
            ORDER BY m.created DESC""",
        (pat, pat)).fetchall()
    c.close()
    seen: dict[str, dict] = {}
    for tid, title, created, content in rows:
        if tid in seen:
            continue
        idx = content.lower().find(q.lower())
        if idx < 0:
            snippet = (title or "")[:120]
        else:
            start = max(0, idx - 40)
            snippet = content[start:idx + len(q) + 80].replace("\n", " ")
            snippet = ("..." if start > 0 else "") + snippet
        seen[tid] = {"id": tid, "title": title, "created": created,
                     "snippet": snippet}
        if len(seen) >= limit:
            break
    return list(seen.values())
