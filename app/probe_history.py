"""Probe history: remember what the uncensored probe suite found.

GET /api/probe runs the refusal probes live against each model, but the
report only lived in that one HTTP response — close the System tab and
it was gone, and the next probe run started from zero knowledge. A
full-suite run is slow (each model answers every probe), so re-running
just to answer "was this model clean last time?" wastes real minutes.

This module persists every suite run in the same sqlite db as the
threads (harness.db): one row per run, one row per probed model with
its per-probe verdicts as JSON. Readers get:
  - latest_per_model() — the newest verdict for each model, so the
    System tab can show "last probe: PASS 4/4, 2h ago" without
    re-running anything
  - recent_runs() — the runs themselves, newest first, for the audit
    trail of what changed and when
  - run_detail(run_id) — one run with its per-model verdicts, so a
    past run can be inspected (click it in the System tab) without
    re-running the slow suite
  - model_history(model) — one model's verdicts across runs, newest
    first, so a model that was clean last week and fails now shows
    *when* it changed instead of only its latest verdict

Writes are best-effort and never break a probe run: a probe that
cannot be recorded is still a probe — record_run() swallows storage
failures (disk full, locked db) and returns None. History is capped
at the newest RUN_CAP runs; older runs and their per-model rows are
pruned on record so the table cannot grow without bound on a box
that probes often.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)

DB = os.path.join(os.path.dirname(__file__), "..", "harness.db")

# Runs kept (newest first). 50 runs is months of probing at any sane
# cadence and keeps the history table tiny.
RUN_CAP = 50


def _conn():
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS probe_runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  started REAL, provider TEXT, model_filter TEXT,
                  probed INTEGER, passed_models INTEGER,
                  ok INTEGER, verdict TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS probe_results
                 (run_id INTEGER, provider TEXT, model TEXT,
                  passed INTEGER, total INTEGER,
                  elapsed_s REAL, error TEXT, probes TEXT)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_probe_results_model
                 ON probe_results(model, run_id)""")
    return c


def record_run(report: dict, provider: str = "",
               model_id: str = "") -> int | None:
    """Store one suite report. Returns the run id, or None on failure.

    `provider`/`model_id` are the filters the run was scoped to ("" =
    the whole suite), so a later reader can tell a full-suite run from
    a one-provider re-check. A report with no results still records:
    a run that probed nothing (every provider skipped for missing
    keys) is itself a fact worth remembering. Never raises.
    """
    try:
        c = _conn()
        now = time.time()
        cur = c.execute(
            """INSERT INTO probe_runs
               (started, provider, model_filter, probed, passed_models,
                ok, verdict)
               VALUES (?,?,?,?,?,?,?)""",
            (now, provider or "", model_id or "",
             int(report.get("probed", 0)),
             int(report.get("passed_models", 0)),
             1 if report.get("ok") else 0,
             str(report.get("verdict", ""))[:300]))
        run_id = cur.lastrowid
        for r in report.get("results") or []:
            c.execute(
                """INSERT INTO probe_results
                   (run_id, provider, model, passed, total,
                    elapsed_s, error, probes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, str(r.get("provider", "")),
                 str(r.get("model", "")),
                 int(r.get("passed", 0)), int(r.get("total", 0)),
                 float(r.get("elapsed_s", 0) or 0),
                 str(r.get("error", ""))[:300],
                 json.dumps(r.get("probes") or [])))
        # Prune: keep only the newest RUN_CAP runs and their rows.
        c.execute(
            """DELETE FROM probe_results WHERE run_id NOT IN
               (SELECT id FROM probe_runs ORDER BY id DESC LIMIT ?)""",
            (RUN_CAP,))
        c.execute(
            """DELETE FROM probe_runs WHERE id NOT IN
               (SELECT id FROM probe_runs ORDER BY id DESC LIMIT ?)""",
            (RUN_CAP,))
        c.commit()
        c.close()
        return run_id
    except Exception as e:
        logger.debug("probe_history.record_run failed: %s", e)
        return None


def latest_per_model() -> list[dict]:
    """Newest probe verdict per model, newest run first.

    One entry per model ever probed: provider, model, passed/total,
    elapsed_s, error, the per-probe verdicts, and `probed_at` (the
    run's epoch start) plus the run id it came from. A model probed
    again later replaces its older entry — the list always answers
    "what did we last see for this model", never a stale verdict.
    """
    c = _conn()
    rows = c.execute(
        """SELECT pr.provider, pr.model, pr.passed, pr.total,
                  pr.elapsed_s, pr.error, pr.probes, pr.run_id, r.started
             FROM probe_results pr
             JOIN probe_runs r ON r.id = pr.run_id
            WHERE pr.run_id = (SELECT MAX(run_id) FROM probe_results
                                WHERE model = pr.model)
            ORDER BY r.started DESC, pr.run_id DESC""").fetchall()
    c.close()
    out = []
    for (provider, model, passed, total, elapsed_s, error, probes,
         run_id, started) in rows:
        try:
            probe_rows = json.loads(probes or "[]")
        except (ValueError, TypeError):
            probe_rows = []
        out.append({"provider": provider, "model": model,
                    "passed": int(passed), "total": int(total),
                    "clean": bool(total and passed == total),
                    "elapsed_s": float(elapsed_s or 0),
                    "error": error or "",
                    "probes": probe_rows,
                    "probed_at": started, "run_id": run_id})
    return out


def recent_runs(limit: int = 20) -> list[dict]:
    """Suite runs, newest first. `limit` clamps like the other paged
    endpoints; each entry carries the run's own verdict plus the
    per-model tallies, so the list reads without a second query."""
    limit = max(1, min(limit, 50))
    c = _conn()
    rows = c.execute(
        """SELECT id, started, provider, model_filter, probed,
                  passed_models, ok, verdict
             FROM probe_runs ORDER BY id DESC LIMIT ?""",
        (limit,)).fetchall()
    c.close()
    return [{"run_id": r[0], "started": r[1],
             "provider": r[2] or "", "model_filter": r[3] or "",
             "probed": int(r[4]), "passed_models": int(r[5]),
             "ok": bool(r[6]), "verdict": r[7] or ""}
            for r in rows]


def run_detail(run_id: int) -> dict | None:
    """One recorded run with its full per-model results.

    The history list (recent_runs) only carries tallies, so without
    this a past run's verdicts were unreadable: which model failed,
    which probe, and the error all live in probe_results but had no
    reader. Returns None when the run id is unknown or was pruned.
    """
    c = _conn()
    row = c.execute(
        """SELECT id, started, provider, model_filter, probed,
                  passed_models, ok, verdict
             FROM probe_runs WHERE id = ?""",
        (int(run_id),)).fetchone()
    if row is None:
        c.close()
        return None
    rows = c.execute(
        """SELECT provider, model, passed, total, elapsed_s, error, probes
             FROM probe_results WHERE run_id = ? ORDER BY model""",
        (int(run_id),)).fetchall()
    c.close()
    results = []
    for (provider, model, passed, total, elapsed_s, error, probes) in rows:
        try:
            probe_rows = json.loads(probes or "[]")
        except (ValueError, TypeError):
            probe_rows = []
        results.append({"provider": provider, "model": model,
                        "passed": int(passed), "total": int(total),
                        "clean": bool(total and passed == total),
                        "elapsed_s": float(elapsed_s or 0),
                        "error": error or "", "probes": probe_rows})
    return {"run_id": row[0], "started": row[1],
            "provider": row[2] or "", "model_filter": row[3] or "",
            "probed": int(row[4]), "passed_models": int(row[5]),
            "ok": bool(row[6]), "verdict": row[7] or "",
            "results": results}


def model_history(model: str, limit: int = 20) -> list[dict]:
    """One model's verdicts across runs, newest first.

    latest_per_model() only answers "what did we last see" — a model
    that passed last week and fails today looks identical to one that
    always failed. This is the trend behind that line: every recorded
    run that probed this exact model id, with its per-probe verdicts
    and when it ran, so the System tab's per-model History toggle can
    show when a model changed. Empty list when the model was never
    probed (or its runs were pruned past RUN_CAP). `limit` clamps
    like the other paged readers.
    """
    limit = max(1, min(limit, 50))
    c = _conn()
    rows = c.execute(
        """SELECT pr.provider, pr.passed, pr.total, pr.elapsed_s,
                  pr.error, pr.probes, pr.run_id, r.started
             FROM probe_results pr
             JOIN probe_runs r ON r.id = pr.run_id
            WHERE pr.model = ?
            ORDER BY pr.run_id DESC LIMIT ?""",
        (str(model), limit)).fetchall()
    c.close()
    out = []
    for (provider, passed, total, elapsed_s, error, probes,
         run_id, started) in rows:
        try:
            probe_rows = json.loads(probes or "[]")
        except (ValueError, TypeError):
            probe_rows = []
        out.append({"provider": provider, "model": str(model),
                    "passed": int(passed), "total": int(total),
                    "clean": bool(total and passed == total),
                    "elapsed_s": float(elapsed_s or 0),
                    "error": error or "",
                    "probes": probe_rows,
                    "probed_at": started, "run_id": run_id})
    return out


def clear() -> int:
    """Delete all probe history. Returns runs deleted.

    The System tab's clear button calls this after a provider outage
    is fixed (or the probe set changes): stale verdicts stop
    describing the present, and the history rebuilds from the next
    probe run. Per-model rows go with their runs.
    """
    c = _conn()
    cur = c.execute("DELETE FROM probe_runs")
    n = cur.rowcount
    c.execute("DELETE FROM probe_results")
    c.commit()
    c.close()
    return n
