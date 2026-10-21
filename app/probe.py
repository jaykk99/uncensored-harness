"""Per-provider uncensored probe suite.

models.self_test() probes the *active route* (whichever leg answers first).
This module probes each provider *individually*: every model id from
providers.list_models() gets the refusal probes run against it directly,
so Jay can see exactly which providers/models answer clean and which
hedge, refuse, or error.

GATE (standing rule): a provider/model may only be described as
'uncensored' or 'fixed' when it passes ALL probes in this suite. The suite
itself only *reports* — it never flips labels on its own.

Usage:
  from app import probe
  probe.run_suite()                    # every live provider, every model
  probe.run_suite(provider="groq")     # one provider
  probe.run_suite(model_id="horde:koboldcpp/L3-8B-Stheno-v3.2")

CLI:  python -m app.probe [--provider NAME] [--model ID] [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

logger = logging.getLogger(__name__)

from . import models
from . import providers as providers_mod
from .models import SELF_TEST_PROBES, looks_like_refusal


def probe_one_model(provider_name: str, model: str,
                    timeout_s: int = 120) -> dict:
    """Run every refusal probe against a single provider/model.

    Returns a verdict entry:
      { provider, model, passed, total, probes: [...], elapsed_s, error }
    """
    entry: dict = {"provider": provider_name, "model": model,
                   "passed": 0, "total": len(SELF_TEST_PROBES),
                   "probes": [], "elapsed_s": 0.0, "error": ""}
    start = time.time()
    try:
        if provider_name == "horde":
            _probe_via_horde(entry, model, timeout_s)
        elif provider_name == "pollinations":
            _probe_via_pollinations(entry, timeout_s)
        else:
            _probe_via_provider(entry, provider_name, model, timeout_s)
    except Exception as e:  # provider-level failure, not a refusal
        entry["error"] = str(e)[:300]
        logger.warning("probe %s/%s failed: %s", provider_name, model, e)
    entry["elapsed_s"] = round(time.time() - start, 1)
    entry["passed"] = sum(1 for p in entry["probes"] if p["passed"])
    if entry["passed"] == 0 and not entry["error"]:
        # Every probe errored — surface the first failure at the top level
        # too so callers don't have to dig through per-probe details.
        errs = [p["detail"] for p in entry["probes"]
                if p["detail"].startswith("ERROR")]
        if errs and len(errs) == len(entry["probes"]):
            entry["error"] = errs[0]
    return entry


def _run_probe_text(prompt: str, send) -> tuple[str, str]:
    """send(prompt) -> raw text. Returns (text, detail)."""
    text = send(prompt)
    text = (text or "").strip()
    if not text:
        return "", "ERROR: empty response"
    if looks_like_refusal(text):
        return text, "REFUSAL detected: " + text[:220]
    return text, text[:220]


def _probe_via_horde(entry: dict, model: str, timeout_s: int) -> None:
    # Horde model ids are bare ("koboldcpp/..."); the list_models id has a
    # "horde:" prefix — strip it for preferred_model.
    bare = model.split("horde:", 1)[-1]
    for p in SELF_TEST_PROBES:
        row = {"id": p["id"], "label": p["label"],
               "passed": False, "detail": ""}
        try:
            text = models.horde_chat(
                [{"role": "user", "content": p["prompt"]}],
                max_tokens=300, temperature=0.7, timeout_s=timeout_s,
                leg="probe", preferred_model=bare)
            _, detail = _run_probe_text(p["prompt"], lambda _pr: text)
            row["detail"] = detail
            row["passed"] = not detail.startswith(("REFUSAL", "ERROR"))
        except Exception as e:
            row["detail"] = f"ERROR: {e}"[:300]
        entry["probes"].append(row)


def _probe_via_pollinations(entry: dict, timeout_s: int) -> None:
    for p in SELF_TEST_PROBES:
        row = {"id": p["id"], "label": p["label"],
               "passed": False, "detail": ""}
        try:
            msg = models.pollinations_chat(
                [{"role": "user", "content": p["prompt"]}],
                max_tokens=300, temperature=0.7, timeout=timeout_s,
                retries=2)
            _, detail = _run_probe_text(p["prompt"],
                                        lambda _pr: msg.get("content", ""))
            row["detail"] = detail
            row["passed"] = not detail.startswith(("REFUSAL", "ERROR"))
        except Exception as e:
            row["detail"] = f"ERROR: {e}"[:300]
        entry["probes"].append(row)


def _probe_via_provider(entry: dict, provider_name: str, model: str,
                        timeout_s: int) -> None:
    p = providers_mod.get_provider(provider_name)
    if p is None:
        raise RuntimeError(f"unknown provider {provider_name}")
    bare = model.split(f"{provider_name}:", 1)[-1]
    for pr in SELF_TEST_PROBES:
        row = {"id": pr["id"], "label": pr["label"],
               "passed": False, "detail": ""}
        try:
            msg = providers_mod.call_provider(
                p, bare, [{"role": "user", "content": pr["prompt"]}],
                max_tokens=300, temperature=0.7, timeout=timeout_s)
            _, detail = _run_probe_text(pr["prompt"],
                                        lambda _pr: msg.get("content", ""))
            row["detail"] = detail
            row["passed"] = not detail.startswith(("REFUSAL", "ERROR"))
        except Exception as e:
            row["detail"] = f"ERROR: {e}"[:300]
        entry["probes"].append(row)


def run_suite(provider: str = "", model_id: str = "",
              timeout_s: int = 120) -> dict:
    """Probe every live model (or one provider / one model id).

    Skips providers whose credentials are missing (reported as skipped, not
    failed — a missing key is not a refusal). Returns the full report dict.
    """
    targets: list[tuple[str, str]] = []
    skipped: list[dict] = []
    for m in providers_mod.list_models():
        prov = m["provider"]
        if provider and prov != provider:
            continue
        if model_id and m["id"] != model_id:
            continue
        if not m["live"]:
            skipped.append({"id": m["id"], "reason": "no credential set"})
            continue
        targets.append((prov, m["id"]))

    results = [probe_one_model(prov, mid, timeout_s) for prov, mid in targets]
    passed_models = sum(1 for r in results if r["passed"] == r["total"])
    return {
        "ok": bool(results) and passed_models == len(results),
        "probed": len(results),
        "passed_models": passed_models,
        "skipped": skipped,
        "verdict": ("ALL CLEAR — every live model answered all probes"
                    if results and passed_models == len(results)
                    else f"{passed_models}/{len(results)} models fully clean"),
        "results": results,
    }


def summarize(report: dict) -> str:
    """One-line-per-model human summary of a suite report."""
    lines = [f"probe suite: {report['verdict']}"]
    for r in report["results"]:
        mark = "PASS" if r["passed"] == r["total"] else "FAIL"
        extra = f" ({r['error']})" if r["error"] else ""
        lines.append(f"  [{mark}] {r['model']} "
                     f"{r['passed']}/{r['total']}{extra}")
    for s in report["skipped"]:
        lines.append(f"  [SKIP] {s['id']} — {s['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Uncensored provider probe suite")
    ap.add_argument("--provider", default="", help="only this provider")
    ap.add_argument("--model", default="", help="only this model id")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = run_suite(provider=args.provider, model_id=args.model,
                       timeout_s=args.timeout)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(summarize(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
