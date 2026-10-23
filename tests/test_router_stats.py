"""Tests for app/router_stats.py — leg health tracking.

DB path is module-level; tests monkeypatch it to a tmp file.
"""

from __future__ import annotations

import pytest

from app import router_stats as rs


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "DB", str(tmp_path / "stats.db"))


def test_record_and_snapshot_ok():
    rs.record("horde", "stheno", "ok", 1200)
    rs.record("horde", "stheno", "ok", 800)
    (row,) = rs.snapshot()
    assert row["provider"] == "horde" and row["model"] == "stheno"
    assert row["attempts"] == 2 and row["successes"] == 2
    assert row["success_rate"] == 1.0
    assert row["avg_ms"] == 1000
    assert row["refusals"] == 0 and row["errors"] == 0
    assert row["last_ok"] is not None


def test_record_refusal_and_error():
    rs.record("kilo", "m1", "refusal", 300)
    rs.record("kilo", "m1", "error", 100)
    (row,) = rs.snapshot()
    assert row["attempts"] == 2 and row["successes"] == 0
    assert row["refusals"] == 1 and row["errors"] == 1
    assert row["success_rate"] == 0.0
    assert row["last_ok"] is None


def test_record_rejects_bad_outcome():
    with pytest.raises(ValueError, match="bad outcome"):
        rs.record("x", "y", "maybe", 10)


def test_record_never_raises_on_db_failure(monkeypatch):
    monkeypatch.setattr(rs, "DB", "/nonexistent-dir-xyz/stats.db")
    # _conn makedirs would actually create it; force failure instead.
    monkeypatch.setattr(rs, "_conn", lambda: (_ for _ in ()).throw(OSError("x")))
    rs.record("h", "m", "ok", 1)  # must not raise


def test_models_tracked_separately():
    rs.record("horde", "a", "ok", 100)
    rs.record("horde", "b", "error", 100)
    rows = {r["model"]: r for r in rs.snapshot()}
    assert rows["a"]["success_rate"] == 1.0
    assert rows["b"]["success_rate"] == 0.0


def test_health_rollup():
    rs.record("horde", "a", "ok", 100)
    rs.record("horde", "a", "refusal", 100)
    rs.record("kilo", "m", "error", 50)
    h = rs.health()
    assert h["legs"] == 2
    assert h["total_attempts"] == 3
    assert h["providers"]["horde"]["models"] == 1
    assert h["providers"]["horde"]["success_rate"] == 0.5
    assert h["providers"]["kilo"]["errors"] == 1


def test_reset():
    rs.record("horde", "a", "ok", 100)
    rs.record("kilo", "m", "ok", 100)
    assert rs.reset(provider="horde") == 1
    assert [r["provider"] for r in rs.snapshot()] == ["kilo"]
    assert rs.reset() == 1
    assert rs.snapshot() == []


def test_snapshot_empty():
    assert rs.snapshot() == []
    assert rs.health() == {"providers": {}, "legs": 0, "stale_legs": 0,
                           "total_attempts": 0}


# ---------------------------------------------------------------------------
# Latency percentiles (p50/p95 from capped per-leg samples)
# ---------------------------------------------------------------------------

def test_percentiles_nearest_rank():
    for i in range(1, 101):  # latencies 1..100 ms
        rs.record("kilo", "m", "ok", i)
    (row,) = rs.snapshot()
    assert row["p50_ms"] == 50
    assert row["p95_ms"] == 95
    assert row["avg_ms"] == 50  # avg rounds down here; p95 tells the real story


def test_percentiles_odd_sample_count():
    for ms in (10, 20, 30, 40, 50):
        rs.record("kilo", "m", "ok", ms)
    (row,) = rs.snapshot()
    assert row["p50_ms"] == 30
    assert row["p95_ms"] == 50


def test_percentiles_separate_per_leg():
    rs.record("kilo", "fast", "ok", 10)
    rs.record("kilo", "slow", "ok", 900)
    rows = {r["model"]: r for r in rs.snapshot()}
    assert rows["fast"]["p95_ms"] == 10
    assert rows["slow"]["p95_ms"] == 900


def test_latency_samples_capped():
    for i in range(rs.LAT_SAMPLE_CAP + 50):
        rs.record("kilo", "m", "ok", i)
    from app import router_stats as rs_mod
    c = rs_mod._conn()
    n = c.execute("SELECT COUNT(*) FROM router_lat").fetchone()[0]
    c.close()
    assert n == rs.LAT_SAMPLE_CAP
    # Percentile is computed over the newest samples (0..49 pruned away).
    (row,) = rs_mod.snapshot()
    assert row["p50_ms"] > 100


def test_reset_clears_latency_samples():
    from app import router_stats as rs_mod
    rs_mod.record("kilo", "m", "ok", 10)
    assert rs_mod.reset(provider="kilo") == 1
    c = rs_mod._conn()
    n = c.execute("SELECT COUNT(*) FROM router_lat").fetchone()[0]
    c.close()
    assert n == 0
    # Re-recorded leg starts with fresh percentiles, no ghost history.
    rs_mod.record("kilo", "m", "ok", 500)
    (row,) = rs_mod.snapshot()
    assert row["p50_ms"] == 500 and row["p95_ms"] == 500


# ---------------------------------------------------------------------------
# Stale legs (no data within ROUTER_STALE_HOURS, default 24)
# ---------------------------------------------------------------------------

def test_fresh_leg_not_stale():
    rs.record("kilo", "m", "ok", 10)
    (row,) = rs.snapshot()
    assert row["stale"] is False


def test_old_leg_is_stale():
    import time
    from app import router_stats as rs_mod
    rs_mod.record("kilo", "m", "ok", 10)
    c = rs_mod._conn()
    c.execute("UPDATE router_stats SET last_seen=?",
              (time.time() - 25 * 3600,))
    c.commit()
    c.close()
    (row,) = rs_mod.snapshot()
    assert row["stale"] is True


def test_stale_threshold_env_override(monkeypatch):
    import time
    from app import router_stats as rs_mod
    monkeypatch.setenv("ROUTER_STALE_HOURS", "1")
    rs_mod.record("kilo", "m", "ok", 10)
    c = rs_mod._conn()
    c.execute("UPDATE router_stats SET last_seen=?",
              (time.time() - 2 * 3600,))
    c.commit()
    c.close()
    (row,) = rs_mod.snapshot()
    assert row["stale"] is True


def test_health_counts_stale_legs():
    import time
    from app import router_stats as rs_mod
    rs_mod.record("kilo", "fresh", "ok", 10)
    rs_mod.record("kilo", "old", "ok", 10)
    c = rs_mod._conn()
    c.execute("UPDATE router_stats SET last_seen=? WHERE model='old'",
              (time.time() - 30 * 3600,))
    c.commit()
    c.close()
    h = rs_mod.health()
    assert h["legs"] == 2
    assert h["stale_legs"] == 1
