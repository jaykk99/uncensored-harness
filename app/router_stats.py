"""Router statistics: per-provider/model health tracking.

Every leg of models.chat() records its outcome here (success, refusal, or
error + latency). The data powers:
  - GET /api/router-stats — which legs are healthy right now
  - the chat UI's provider health dots
  - routing: omni_chain() demotes legs with poor recent stats within
    their tier (success <50% over >=3 attempts, or stale data)

Beyond the aggregate counters, each record keeps a capped sample of the
leg's recent latencies (router_lat table, newest 200 per leg). That gives
the snapshot real p50/p95 tail-latency percentiles — an average alone
hides the slow outliers that actually hurt a turn. Legs that haven't been
seen in ROUTER_STALE_HOURS (default 24) are flagged `stale` so the UI can
grey out legs whose numbers no longer describe the present.

An error count alone never says *why* a leg is failing — a 429, a bad
key, a timeout all look identical in the health table. So each leg also
keeps its most recent failure reason (`last_error`, truncated, with
`last_error_at`): the System tab shows it next to the error count, so a
dead leg can be diagnosed without server logs. A later success clears
it — a recovered leg must not keep showing a stale failure.

Storage is a tiny sqlite db next to the thread db (harness.db). Writes are
best-effort and never break a chat turn.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DB = os.path.join(os.path.dirname(__file__), "..", "harness.db")

# Per-leg latency samples kept (newest first). 200 is plenty for stable
# percentiles and keeps the db tiny.
LAT_SAMPLE_CAP = 200

# Last-error text is a diagnostic, not a log: one short line per leg.
LAST_ERROR_CAP = 300


def _stale_after_s() -> float:
    try:
        hours = float(os.environ.get("ROUTER_STALE_HOURS", "24"))
    except ValueError:
        hours = 24.0
    return max(1.0, hours) * 3600.0


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
                  last_error TEXT, last_error_at REAL,
                  PRIMARY KEY (provider, model))""")
    # Older DBs were created before the last_error columns existed: add
    # them in place so a deployed harness keeps its stats instead of
    # crashing every record call (same pattern as memory.py's columns).
    cols = {r[1] for r in c.execute("PRAGMA table_info(router_stats)")}
    if "last_error" not in cols:
        c.execute("ALTER TABLE router_stats ADD COLUMN last_error TEXT")
    if "last_error_at" not in cols:
        c.execute("ALTER TABLE router_stats ADD COLUMN last_error_at REAL")
    c.execute("""CREATE TABLE IF NOT EXISTS router_lat
                 (provider TEXT, model TEXT, ms INTEGER, ts REAL)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_router_lat_leg
                 ON router_lat(provider, model, ts)""")
    return c


def record(provider: str, model: str, outcome: str, ms: int,
           detail: str = "") -> None:
    """Record one leg outcome. outcome in {"ok","refusal","error"}.

    detail is the failure reason (only used on "error"): the exception
    text from the leg, truncated to LAST_ERROR_CAP. A later "ok" clears
    it, so a recovered leg stops showing a stale failure; a "refusal"
    leaves it alone (a refusal is neither a recovery nor an error), and
    an error with no detail keeps the previous reason instead of
    wiping it with nothing.
    """
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
        err_text = (detail or "").strip()[:LAST_ERROR_CAP] or None
        err_val = err_text if outcome == "error" else None
        err_ts = now if outcome == "error" and err_text else None
        c.execute(
            """INSERT INTO router_stats
               (provider, model, attempts, successes, refusals, errors,
                total_ms, last_ok, last_seen, last_error, last_error_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider, model) DO UPDATE SET
                 attempts = attempts + 1,
                 successes = successes + ?,
                 refusals = refusals + ?,
                 errors = errors + ?,
                 total_ms = total_ms + ?,
                 last_ok = CASE WHEN ? THEN ? ELSE last_ok END,
                 last_seen = ?,
                 last_error = CASE WHEN ? THEN NULL
                                   WHEN ? AND ? IS NOT NULL THEN ?
                                   ELSE last_error END,
                 last_error_at = CASE WHEN ? THEN NULL
                                      WHEN ? AND ? IS NOT NULL THEN ?
                                      ELSE last_error_at END""",
            # 11 VALUES placeholders + 15 ON CONFLICT placeholders.
            (provider, model, 1, ok_d, ref_d, err_d, ms_int, last_ok_val, now,
             err_val, err_ts,
             ok_d, ref_d, err_d, ms_int, ok_d, last_ok_val, now,
             ok_d, err_d, err_val, err_val,
             ok_d, err_d, err_ts, err_ts))
        c.execute("INSERT INTO router_lat (provider, model, ms, ts) "
                  "VALUES (?,?,?,?)", (provider, model, ms_int, now))
        # Prune: keep only the newest LAT_SAMPLE_CAP samples per leg.
        c.execute(
            """DELETE FROM router_lat WHERE provider=? AND model=?
               AND rowid NOT IN
               (SELECT rowid FROM router_lat WHERE provider=? AND model=?
                ORDER BY ts DESC, rowid DESC LIMIT ?)""",
            (provider, model, provider, model, LAT_SAMPLE_CAP))
        c.commit()
        c.close()
    except Exception as e:
        logger.debug("router_stats.record failed: %s", e)


def _percentile(sorted_ms: list[int], pct: float) -> int:
    """Nearest-rank percentile on an ascending-sorted sample list."""
    if not sorted_ms:
        return 0
    rank = math.ceil(pct / 100.0 * len(sorted_ms)) - 1
    return int(sorted_ms[max(0, min(rank, len(sorted_ms) - 1))])


def snapshot() -> list[dict]:
    """Per-leg stats with derived success rate, latency and staleness."""
    c = _conn()
    rows = c.execute(
        """SELECT provider, model, attempts, successes, refusals, errors,
                  total_ms, last_ok, last_seen, last_error, last_error_at
           FROM router_stats ORDER BY successes DESC, attempts DESC""").fetchall()
    lat_rows = c.execute(
        "SELECT provider, model, ms FROM router_lat").fetchall()
    c.close()
    lat: dict[tuple[str, str], list[int]] = {}
    for provider, model, ms in lat_rows:
        lat.setdefault((provider, model), []).append(int(ms))
    stale_after = _stale_after_s()
    now = time.time()
    out = []
    for provider, model, attempts, ok, refusals, errors, total_ms, last_ok, last_seen, last_error, last_error_at in rows:
        samples = sorted(lat.get((provider, model), []))
        avg = int(total_ms / attempts) if attempts else 0
        out.append({
            "provider": provider, "model": model,
            "attempts": attempts, "successes": ok,
            "refusals": refusals, "errors": errors,
            "success_rate": round(ok / attempts, 3) if attempts else 0.0,
            "avg_ms": avg,
            "p50_ms": _percentile(samples, 50) if samples else avg,
            "p95_ms": _percentile(samples, 95) if samples else avg,
            "stale": bool(last_seen is None or now - last_seen > stale_after),
            "last_ok": last_ok, "last_seen": last_seen,
            "last_error": last_error, "last_error_at": last_error_at,
        })
    return out


def reset(provider: str = "", model: str = "") -> int:
    """Clear stats (all, or one leg). Returns rows deleted.

    Latency samples for the cleared legs are deleted too, so the
    percentiles rebuild from fresh data alongside the counters.
    """
    c = _conn()
    if provider and model:
        cur = c.execute("DELETE FROM router_stats WHERE provider=? AND model=?",
                        (provider, model))
        c.execute("DELETE FROM router_lat WHERE provider=? AND model=?",
                  (provider, model))
    elif provider:
        cur = c.execute("DELETE FROM router_stats WHERE provider=?", (provider,))
        c.execute("DELETE FROM router_lat WHERE provider=?", (provider,))
    else:
        cur = c.execute("DELETE FROM router_stats")
        c.execute("DELETE FROM router_lat")
    n = cur.rowcount
    c.commit()
    c.close()
    return n


def health() -> dict:
    """Rollup for the UI: per-provider totals + overall."""
    snap = snapshot()
    by_provider: dict[str, dict] = {}
    stale_legs = 0
    for row in snap:
        if row["stale"]:
            stale_legs += 1
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
            "stale_legs": stale_legs,
            "total_attempts": total_attempts}
