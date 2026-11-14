"""Training data: real labeled prompts from tests/test_guards.py.

The guard test suite asserts a verdict for each prompt it lists:
HARD_CASES + the crafting / carve-out-cancel cases (label 1 =
blocked by the hard boundary) and LEGIT_CASES + the analysis /
empty cases (label 0 = passes). This loader parses that test file
with ast and collects exactly those prompts: the first element of
each HARD_CASES tuple, each LEGIT_CASES string, and — inside the
named test functions — the strings passed to check_hard_boundary()
and the strings in list literals those functions loop over.
Assertion substrings (e.g. "phishing" in hit.reason) are NOT
collected: they are not prompts. Nothing is invented and no verdict
is re-labeled: the label is the verdict the test itself asserts.

That is the whole labeled set the repo has: the harness keeps no
run logs with outcomes (harness.db holds 10 threads / 17 messages
with no success/failure recorded, and router_stats is empty), so
there is no larger labeled source to train on. The small size is
reported honestly with the metrics, not hidden.
"""
from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Dict, List, Tuple

LABEL_BLOCKED = 1
LABEL_ALLOWED = 0

TEST_FILE = Path(__file__).resolve().parents[1] / "tests" / "test_guards.py"

# Test functions whose prompts are label 1 / label 0. The parametrize
# users (test_each_hard_boundary_fires, test_legit_work_passes) are
# covered by the HARD_CASES / LEGIT_CASES lists instead.
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


def _func_prompts(fn: ast.FunctionDef) -> List[str]:
    """Prompts inside one test function: check_hard_boundary() args
    plus strings in list literals it loops over."""
    out: List[str] = []
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


def build_dataset(test_path: str | Path = TEST_FILE) -> Dict:
    """Build the labeled training set. Returns rows + build stats."""
    tree = ast.parse(Path(test_path).read_text(encoding="utf-8"))
    blocked: List[str] = []
    allowed: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "HARD_CASES":
                for elt in node.value.elts:
                    blocked.append(ast.literal_eval(elt.elts[0]))
            elif name == "LEGIT_CASES":
                for elt in node.value.elts:
                    allowed.append(ast.literal_eval(elt))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in _BLOCKED_FUNCS:
            blocked.extend(_func_prompts(node))
        elif node.name in _ALLOWED_FUNCS:
            allowed.extend(_func_prompts(node))
    rows: List[Dict] = []
    seen = set()
    dupes = 0
    for text, label in [(t, LABEL_BLOCKED) for t in blocked] + \
            [(t, LABEL_ALLOWED) for t in allowed]:
        if text in seen:
            dupes += 1
            continue
        seen.add(text)
        rows.append({"text": text, "label": label})
    stats = {
        "source": "tests/test_guards.py HARD_CASES / LEGIT_CASES plus "
                  "the crafting, carve-out, analysis and empty-input "
                  "test functions",
        "blocked_raw": len(blocked),
        "allowed_raw": len(allowed),
        "duplicate_prompts_dropped": dupes,
        "train_rows": len(rows),
        "n_positive": sum(1 for r in rows if r["label"] == 1),
        "n_negative": sum(1 for r in rows if r["label"] == 0),
    }
    return {"rows": rows, "stats": stats}


def stratified_split(rows: List[Dict], test_frac: float = 0.2,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/test split (keeps the class ratio in both)."""
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train: List[Dict] = []
    test: List[Dict] = []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac)) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
