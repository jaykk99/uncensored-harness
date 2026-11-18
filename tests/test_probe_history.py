"""Tests for app/probe_history.py — persisted probe verdicts.

Storage mirrors router_stats: the module-level DB path is monkeypatched
to a tmp file, so no test writes into the repo's real harness.db.
"""

from __future__ import annotations

import pytest

from app import probe_history as hist


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(hist, "DB", str(tmp_path / "probe.db"))
    return hist


def _report(model="llm7:codestral-latest", passed=4, total=4,
            error="", verdict="ALL CLEAR — every live model answered all probes"):
    return {"ok": passed == total, "probed": 1, "passed_models": 1 if passed == total else 0,
            "skipped": [], "verdict": verdict,
            "results": [{"provider": "llm7", "model": model,
                         "passed": passed, "total": total,
                         "probes": [{"id": "hobby_dual_use", "label": "Dual-use",
                                     "passed": passed == total, "detail": "ok"}],
                         "elapsed_s": 1.5, "error": error}]}


def test_record_and_latest_round_trip(tmpdb):
    run_id = tmpdb.record_run(_report())
    assert run_id is not None
    latest = tmpdb.latest_per_model()
    assert len(latest) == 1
    row = latest[0]
    assert row["model"] == "llm7:codestral-latest"
    assert row["passed"] == 4 and row["total"] == 4
    assert row["clean"] is True
    assert row["run_id"] == run_id
    assert row["probed_at"] > 0
    assert row["probes"][0]["id"] == "hobby_dual_use"
    runs = tmpdb.recent_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["ok"] is True
    assert runs[0]["probed"] == 1 and runs[0]["passed_models"] == 1


def test_latest_keeps_only_newest_verdict_per_model(tmpdb):
    tmpdb.record_run(_report(passed=0, total=4, error="boom",
                             verdict="0/1 models fully clean"))
    tmpdb.record_run(_report(passed=4, total=4))
    latest = tmpdb.latest_per_model()
    assert len(latest) == 1
    assert latest[0]["clean"] is True
    assert latest[0]["error"] == ""


def test_latest_keeps_other_models(tmpdb):
    tmpdb.record_run(_report(model="llm7:model-a", passed=4))
    other = _report(model="kilo:kilo-auto/free", passed=2, total=4,
                    verdict="1/2 models fully clean")
    other["probed"] = 1
    other["passed_models"] = 0
    other["ok"] = False
    tmpdb.record_run(other)
    latest = tmpdb.latest_per_model()
    assert {r["model"] for r in latest} == {"llm7:model-a", "kilo:kilo-auto/free"}
    by_model = {r["model"]: r for r in latest}
    assert by_model["llm7:model-a"]["clean"] is True
    assert by_model["kilo:kilo-auto/free"]["clean"] is False
    assert by_model["kilo:kilo-auto/free"]["passed"] == 2


def test_empty_history(tmpdb):
    assert tmpdb.latest_per_model() == []
    assert tmpdb.recent_runs() == []


def test_empty_report_still_records_a_run(tmpdb):
    run_id = tmpdb.record_run({"ok": False, "probed": 0, "passed_models": 0,
                               "skipped": [], "verdict": "0/0 models fully clean",
                               "results": []})
    assert run_id is not None
    assert tmpdb.latest_per_model() == []
    runs = tmpdb.recent_runs()
    assert len(runs) == 1 and runs[0]["probed"] == 0


def test_recent_runs_newest_first_and_limit(tmpdb):
    for i in range(5):
        tmpdb.record_run(_report(model=f"llm7:m{i}"))
    runs = tmpdb.recent_runs(limit=2)
    assert len(runs) == 2
    assert runs[0]["run_id"] > runs[1]["run_id"]
    assert runs[0]["started"] >= runs[1]["started"]


def test_run_cap_prunes_old_runs(tmpdb, monkeypatch):
    monkeypatch.setattr(hist, "RUN_CAP", 3)
    for i in range(5):
        tmpdb.record_run(_report(model=f"llm7:m{i}"))
    runs = tmpdb.recent_runs(limit=50)
    assert len(runs) == 3
    latest = tmpdb.latest_per_model()
    assert {r["model"] for r in latest} == {"llm7:m2", "llm7:m3", "llm7:m4"}


def test_run_detail_serves_full_results(tmpdb):
    report = _report(passed=2, total=4, error="flaky leg",
                     verdict="0/1 models fully clean")
    report["results"].append(
        {"provider": "kilo", "model": "kilo:kilo-auto/free",
         "passed": 4, "total": 4,
         "probes": [{"id": "hobby_dual_use", "label": "Dual-use",
                     "passed": True, "detail": "ok"}],
         "elapsed_s": 2.5, "error": ""})
    report["probed"] = 2
    run_id = tmpdb.record_run(report)
    d = tmpdb.run_detail(run_id)
    assert d is not None
    assert d["run_id"] == run_id
    assert d["verdict"] == "0/1 models fully clean"
    assert d["probed"] == 2 and d["passed_models"] == 0
    assert d["ok"] is False
    assert [r["model"] for r in d["results"]] == [
        "kilo:kilo-auto/free", "llm7:codestral-latest"]
    by_model = {r["model"]: r for r in d["results"]}
    bad = by_model["llm7:codestral-latest"]
    assert bad["passed"] == 2 and bad["total"] == 4
    assert bad["clean"] is False and bad["error"] == "flaky leg"
    assert bad["probes"][0]["id"] == "hobby_dual_use"
    good = by_model["kilo:kilo-auto/free"]
    assert good["clean"] is True and good["elapsed_s"] == 2.5


def test_run_detail_unknown_and_empty(tmpdb):
    assert tmpdb.run_detail(999) is None
    run_id = tmpdb.record_run({"ok": False, "probed": 0, "passed_models": 0,
                               "skipped": [], "verdict": "0/0 models fully clean",
                               "results": []})
    d = tmpdb.run_detail(run_id)
    assert d is not None and d["results"] == []


def test_run_detail_pruned_run_is_gone(tmpdb, monkeypatch):
    monkeypatch.setattr(hist, "RUN_CAP", 1)
    first = tmpdb.record_run(_report(model="llm7:m0"))
    tmpdb.record_run(_report(model="llm7:m1"))
    assert tmpdb.run_detail(first) is None


def test_clear_wipes_runs_and_results(tmpdb):
    tmpdb.record_run(_report())
    tmpdb.record_run(_report(model="kilo:kilo-auto/free"))
    assert tmpdb.clear() == 2
    assert tmpdb.latest_per_model() == []
    assert tmpdb.recent_runs() == []
    assert tmpdb.clear() == 0


def test_record_never_raises_on_db_failure(monkeypatch):
    monkeypatch.setattr(hist, "_conn",
                        lambda: (_ for _ in ()).throw(OSError("disk full")))
    assert hist.record_run(_report()) is None
    # Reads raise like the sibling modules (callers own the failure);
    # only the write path is best-effort, so a probe run never breaks.
    with pytest.raises(OSError):
        hist.latest_per_model()
