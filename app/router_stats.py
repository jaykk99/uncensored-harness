"""Router statistics: per-provider/model health tracking.

Every leg of models.chat() records its outcome here (success, refusal, or
error + latency). The data powers:
  - GET /api/router-stats — which legs are healthy right now
  - the chat UI's provider health dots
  - future routing: omni_chain() can prefer historically-good legs

Storage is a tiny sqlite db next to the thread db (harness.db). Writes are
best-effort and never break a chat turn.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DB = os.path.join(os.path.dirname(__file__), "..", "harness.db")


def _conn():
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS router_stats
                 (provider TEXT, model TEXT,
                  attempts INTEGER DEFAULT 0,
                  successes INTEGER DEFAULT 0,
                  refusals INTEGER DEFAULT 0,
                  errors INTEGER DEFAULT 0,
                  total_ms INTEGER DEFAULT 0,
                  last_ok REAL, last_seen REAL,
                  PRIMARY KEY (provider, model))""")
    return c


def record(provider: str, model: str, outcome: str, ms: int) -> None:
    """Record one leg outcome. outcome in {"ok","refusal","error"}."""
    if outcome not in ("ok", "refusal", "error"):
        raise ValueError(f"bad outcome: {outcome}")
    try:
        c = _conn()
        now = time.time()
        ok_d = 1 if outcome == "ok" else 0
        ref_d = 1 if outcome == "refusal" else 0
        err_d = 1 if outcome == "error" else 0
        ms_int = max(0, int(ms))
        last_ok_val = now if outcome == "ok" else None
        c.execute(
            """INSERT INTO router_stats
               (provider, model, attempts, successes, refusals, errors,
                total_ms, last_ok, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider, model) DO UPDATE SET
                 attempts = attempts + 1,
                 successes = successes + ?,
                 refusals = refusals + ?,
                 errors = errors + ?,
                 total_ms = total_ms + ?,
                 last_ok = CASE WHEN ? THEN ? ELSE last_ok END,
                 last_seen = ?""",
            # 9 VALUES placeholders + 7 ON CONFLICT placeholders = 16 params.
            (provider, model, 1, ok_d, ref_d, err_d, ms_int, last_ok_val, now,
             ok_d, ref_d, err_d, ms_int, ok_d, last_ok_val, now))
        c.commit()
        c.close()
    except Exception as e:
        logger.debug("router_stats.record failed: %s", e)


def snapshot() -> list[dict]:
    """Per-leg stats with derived success rate and avg latency."""
    c = _conn()
    rows = c.execute(
        """SELECT provider, model, attempts, successes, refusals, errors,
                  total_ms, last_ok, last_seen
           FROM router_stats ORDER BY successes DESC, attempts DESC""").fetchall()
    c.close()
    out = []
    for provider, model, attempts, ok, refusals, errors, total_ms, last_ok, last_seen in rows:
        out.append({
            "provider": provider, "model": model,
            "attempts": attempts, "successes": ok,
            "refusals": refusals, "errors": errors,
            "success_rate": round(ok / attempts, 3) if attempts else 0.0,
            "avg_ms": int(total_ms / attempts) if attempts else 0,
            "last_ok": last_ok, "last_seen": last_seen,
        })
    return out


def reset(provider: str = "", model: str = "") -> int:
    """Clear stats (all, or one leg). Returns rows deleted."""
    c = _conn()
    if provider and model:
        cur = c.execute("DELETE FROM router_stats WHERE provider=? AND model=?",
                        (provider, model))
    elif provider:
        cur = c.execute("DELETE FROM router_stats WHERE provider=?", (provider,))
    else:
        cur = c.execute("DELETE FROM router_stats")
    n = cur.rowcount
    c.commit()
    c.close()
    return n


def health() -> dict:
    """Rollup for the UI: per-provider totals + overall."""
    snap = snapshot()
    by_provider: dict[str, dict] = {}
    for row in snap:
        p = by_provider.setdefault(row["provider"],
                                   {"attempts": 0, "successes": 0,
                                    "refusals": 0, "errors": 0, "models": 0})
        p["attempts"] += row["attempts"]
        p["successes"] += row["successes"]
        p["refusals"] += row["refusals"]
        p["errors"] += row["errors"]
        p["models"] += 1
    for p in by_provider.values():
        p["success_rate"] = (round(p["successes"] / p["attempts"], 3)
                             if p["attempts"] else 0.0)
    total_attempts = sum(p["attempts"] for p in by_provider.values())
    return {"providers": by_provider,
            "legs": len(snap),
            "total_attempts": total_attempts}
