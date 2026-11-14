"""Build the nn-100h dataset: REAL repo prompts only, labeled by the
repo's own guard.

Sources (nothing invented, no synthetic rows):
  1. tests/test_guards.py — HARD_CASES / LEGIT_CASES plus the prompts
     inside the crafting, carve-out-cancel, analysis and empty-input
     test functions (same collection as PR #11, nn/data.py).
  2. tests/test_guard_policy.py — the four prompts asserted against
     check_hard_boundary() plus the CANARY prompt (a complete user
     prompt string shipped in the repo).
  3. app/models.py — SELF_TEST_PROBES: the four user prompts the
     harness itself sends as user messages in its self-test suite.

Labels are NOT copied from test assertions: every collected prompt is
run through the repo's actual app.guards.check_hard_boundary() and
label 1 = blocked, 0 = passes. The script prints any prompt where
the guard verdict disagrees with what its test asserts, so a
disagreement is visible instead of silently re-labeled.

Excluded on purpose (not real user prompts): placeholder strings
("q1", "first question, fixed", "anything at all", "a normal
marketing page"), model replies ("This is not appropriate content."),
refusal phrases, assertion substrings, code, and URLs.

Groups (for the grouped frozen split): each prompt gets a topic
family from its text — card, phishing, wallet, weapon, minor, or
legit_* — so the split can hold out whole families and measure
generalization to unseen keywords instead of memorized phrasing.

Usage (from repo root):
    python3 nn-100h/build_dataset.py
Writes nn-100h/dataset.jsonl, nn-100h/split.json, nn-100h/DATASET.md.
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from app.guards import check_hard_boundary  # noqa: E402  (repo's own guard)

OUT = REPO / "nn-100h"
TEST_GUARDS = REPO / "tests" / "test_guards.py"
TEST_POLICY = REPO / "tests" / "test_guard_policy.py"
MODELS = REPO / "app" / "models.py"

SEED = 42
TEST_FRAC = 0.2

_BLOCKED_FUNCS = {
    "test_phishing_email_crafting_blocked",
    "test_case_insensitive_matching",
    "test_match_inside_longer_sentence",
    "test_defense_carve_outs_do_not_open_crafting",
}
_ALLOWED_FUNCS = {
    "test_phishing_email_analysis_passes",
    "test_empty_and_none_inputs_pass",
}


def _func_prompts(fn: ast.FunctionDef) -> list[str]:
    out: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", "") or getattr(func, "id", "")
            if name == "check_hard_boundary" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    out.append("" if arg.value is None else str(arg.value))
        elif isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(elt.value)
    return out


def collect() -> list[dict]:
    """Return [{text, source, asserted}] in collection order."""
    rows: list[dict] = []

    def add(text: str, source: str, asserted: int | None) -> None:
        if text is None:
            return
        rows.append({"text": text, "source": source, "asserted": asserted})

    tree = ast.parse(TEST_GUARDS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "HARD_CASES":
                for elt in node.value.elts:
                    add(ast.literal_eval(elt.elts[0]),
                        "tests/test_guards.py:HARD_CASES", 1)
            elif name == "LEGIT_CASES":
                for elt in node.value.elts:
                    add(ast.literal_eval(elt),
                        "tests/test_guards.py:LEGIT_CASES", 0)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in _BLOCKED_FUNCS:
            for t in _func_prompts(node):
                add(t, f"tests/test_guards.py:{node.name}", 1)
        elif node.name in _ALLOWED_FUNCS:
            for t in _func_prompts(node):
                add(t, f"tests/test_guards.py:{node.name}", 0)

    # test_guard_policy.py: only the check_hard_boundary() args (the
    # hard-boundary assertions) plus the CANARY user prompt.
    ptree = ast.parse(TEST_POLICY.read_text(encoding="utf-8"))
    for node in ptree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "CANARY":
                add(ast.literal_eval(node.value),
                    "tests/test_guard_policy.py:CANARY", 0)
    for node in ast.walk(ptree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", "") or getattr(func, "id", "")
            if name == "check_hard_boundary" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    add(arg.value, "tests/test_guard_policy.py:hard-boundary", None)

    # app/models.py SELF_TEST_PROBES: prompts the harness sends itself.
    mtree = ast.parse(MODELS.read_text(encoding="utf-8"))
    for node in mtree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "SELF_TEST_PROBES":
                for elt in node.value.elts:
                    d = ast.literal_eval(elt)
                    add(d["prompt"], "app/models.py:SELF_TEST_PROBES", 0)
    return rows


def group_of(text: str, label: int) -> str:
    """Topic family, split by verdict inside a topic: craft vs
    defense is the distinction the guard itself draws (e.g. a
    phishing email you write vs one you filter), so the two sides
    are different families for split purposes."""
    t = (text or "").lower()
    if re.search(r"bioweapon|nuke|dirty bomb", t):
        return "weapon" if label == 1 else "weapon_theme"
    if re.search(r"\bcsam\b|child|minor", t):
        return "minor"
    if "phish" in t or "credential harvest" in t:
        return "phishing_craft" if label == 1 else "phishing_defense"
    if re.search(r"credit card|cvv|stolen card|card dump|dump file", t):
        return "card_abuse" if label == 1 else "card_allowed"
    if re.search(r"wallet|drain|steal.*(crypto|bitcoin|funds)|bitcoin|seed phrase", t):
        return "wallet_abuse" if label == 1 else "wallet_recovery"
    return "legit_general" if label == 0 else "other_blocked"


def grouped_split(rows: list[dict], test_frac: float = TEST_FRAC,
                  seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Grouped stratified split: whole topic families go to train or
    test together (never split across), seed 42. Exhaustive search
    over group subsets (<= ~10 groups) picks the subset closest to
    test_frac of rows with both labels near their stratified
    targets (blocked/allowed each within target +/- 1 and >= 3);
    ties break by seed-shuffled subset order, so the split is
    frozen and reproducible. Test therefore holds out whole
    families and measures generalization to unseen keywords, not
    memorized phrasing. NEVER train on test rows."""
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    groups = sorted(by_group)
    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    n = len(rows)
    target_n = round(n * test_frac)
    n_label = Counter(r["label"] for r in rows)
    target = {lab: max(1, round(c * test_frac)) for lab, c in n_label.items()}
    best: tuple | None = None
    best_ids: set[str] | None = None
    subsets = []
    for mask in range(1, 1 << len(groups)):
        subsets.append(mask)
    rng.shuffle(subsets)
    for mask in subsets:
        ids = set()
        labs = Counter()
        for i in order:
            if mask & (1 << i):
                for r in by_group[groups[i]]:
                    ids.add(r["id"])
                    labs[r["label"]] += 1
        if labs[0] < 3 or labs[1] < 3:
            continue
        if labs[1] > target[1] + 1 or labs[0] > target[0] + 1:
            continue
        score = (abs(len(ids) - target_n),
                 abs(labs[1] - target[1]) + abs(labs[0] - target[0]))
        if best is None or score < best:
            best = score
            best_ids = ids
    assert best_ids is not None, "no grouped split satisfies the targets"
    train = [r for r in rows if r["id"] not in best_ids]
    test = [r for r in rows if r["id"] in best_ids]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main() -> int:
    raw = collect()
    seen: set[str] = set()
    rows: list[dict] = []
    dupes = 0
    mismatches: list[dict] = []
    for i, item in enumerate(raw):
        text = item["text"]
        if text in seen:
            dupes += 1
            continue
        seen.add(text)
        label = 1 if check_hard_boundary(text) is not None else 0
        if item["asserted"] is not None and item["asserted"] != label:
            mismatches.append({"text": text, "asserted": item["asserted"],
                               "guard": label, "source": item["source"]})
        rows.append({"id": f"p{i:03d}", "text": text, "label": label,
                     "source": item["source"],
                     "group": group_of(text, label)})
    rows.sort(key=lambda r: r["id"])
    train, test = grouped_split(rows)
    n_pos = sum(r["label"] for r in rows)

    with open(OUT / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    split = {
        "seed": SEED, "test_frac": TEST_FRAC,
        "method": "grouped stratified: whole topic families (group) go "
                  "to train or test together, seed 42; test holds out "
                  "whole families to measure generalization to unseen "
                  "keywords, not memorized phrasing. NEVER train on test rows.",
        "train_ids": [r["id"] for r in train],
        "test_ids": [r["id"] for r in test],
        "test_groups": sorted({r["group"] for r in test}),
        "train_groups": sorted({r["group"] for r in train}),
    }
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    groups = Counter(r["group"] for r in rows)
    test_groups = Counter(r["group"] for r in test)
    md = f"""# Dataset — nn-100h (uncensored-harness guard triage)

## Provenance
- Collection date: {date.today().isoformat()}
- Sources (real repo data only, nothing invented, no synthetic rows):
  - `tests/test_guards.py` — HARD_CASES / LEGIT_CASES plus the crafting,
    carve-out-cancel, analysis and empty-input test functions
    (same collection as PR #11).
  - `tests/test_guard_policy.py` — the four prompts asserted against
    `check_hard_boundary()` plus the CANARY prompt string.
  - `app/models.py` — SELF_TEST_PROBES, the four user prompts the
    harness itself sends in its self-test suite.
- Label definition: label 1 = the repo's actual
  `app.guards.check_hard_boundary()` returns a hit (blocked);
  label 0 = it returns None (passes). Every prompt was run through
  the real guard at collection time; labels are NOT copied from
  test assertions. Guard/test disagreements found: {len(mismatches)}.
- Excluded on purpose (not real user prompts): placeholder strings
  ("q1", "first question, fixed", "anything at all", "a normal
  marketing page"), model replies, refusal phrases, assertion
  substrings, code, URLs. No larger labeled source exists in the
  repo: the harness keeps no run logs with guard outcomes
  (PR #11 verified: `harness.db` holds threads/messages with no
  success/failure recorded, `router_stats` empty).

## Size
- Rows: {len(rows)} ({n_pos} blocked / {len(rows) - n_pos} allowed)
- Duplicates dropped (exact text): {dupes}
- Train: {len(train)} rows, test: {len(test)} rows
  (test = {len(test) / len(rows):.1%} of rows)

## Groups (topic families)
| group | rows | in test |
|---|---:|---:|
"""
    for g in sorted(groups):
        md += f"| {g} | {groups[g]} | {test_groups.get(g, 0)} |\n"
    md += f"""
## Frozen split
- `split.json`, seed 42, test frac 0.2, GROUPED stratified: whole
  topic families go to train or test together, so the test set
  measures generalization to unseen families/keywords, not
  memorized phrasing. Test groups: {", ".join(split["test_groups"]) or "none"}.
  NEVER train on test rows.
- The test set is {len(test)} prompts, so each test prompt moves
  accuracy by ~{100.0 / max(1, len(test)):.0f} points. Treat test
  numbers as a smoke check, not a benchmark.

## Leakage audit
- The guard is a deterministic rule (5 regex patterns + 3
  carve-outs in `app/guards.py`). Predicting it from prompt text
  is rule reproduction: a model that memorizes the trigger words
  ("phishing kit", "cvv", "dirty bomb") scores well on prompts
  that reuse those words and tells us nothing about unseen
  phrasing. That is why the split is GROUPED — test families are
  absent from training — and why a 1.000 test accuracy must be
  investigated as leakage, never reported as a win.
- Vocab is built from TRAIN rows only; test words unseen in
  training get no word feature, only the generic shape features.
- This model is triage only. `check_hard_boundary()` stays the
  only gate; a low score never means a prompt is safe, and this
  must never be used to bypass, weaken, or replace the guard.
"""
    (OUT / "DATASET.md").write_text(md, encoding="utf-8")
    print(f"rows={len(rows)} blocked={n_pos} allowed={len(rows) - n_pos} "
          f"dupes_dropped={dupes} mismatches={len(mismatches)}")
    print(f"train={len(train)} test={len(test)} "
          f"test_groups={split['test_groups']}")
    print("groups:", dict(sorted(groups.items())))
    for m in mismatches:
        print("MISMATCH:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
