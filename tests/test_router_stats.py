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
    assert rs.health() == {"providers": {}, "legs": 0, "total_attempts": 0}
