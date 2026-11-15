# Dataset — nn-100h (uncensored-harness guard triage)

## Provenance
- Collection date: 2026-11-14
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
  test assertions. Guard/test disagreements found: 0.
- Excluded on purpose (not real user prompts): placeholder strings
  ("q1", "first question, fixed", "anything at all", "a normal
  marketing page"), model replies, refusal phrases, assertion
  substrings, code, URLs. No larger labeled source exists in the
  repo: the harness keeps no run logs with guard outcomes
  (PR #11 verified: `harness.db` holds threads/messages with no
  success/failure recorded, `router_stats` empty).

## Size
- Rows: 53 (23 blocked / 30 allowed)
- Duplicates dropped (exact text): 2
- Train: 43 rows, test: 10 rows
  (test = 18.9% of rows)

## Groups (topic families)
| group | rows | in test |
|---|---:|---:|
| card_abuse | 5 | 0 |
| card_allowed | 3 | 3 |
| legit_general | 16 | 0 |
| minor | 2 | 2 |
| phishing_craft | 11 | 0 |
| phishing_defense | 9 | 0 |
| wallet_abuse | 3 | 3 |
| wallet_recovery | 1 | 1 |
| weapon | 2 | 0 |
| weapon_theme | 1 | 1 |

## Frozen split
- `split.json`, seed 42, test frac 0.2, GROUPED stratified: whole
  topic families go to train or test together, so the test set
  measures generalization to unseen families/keywords, not
  memorized phrasing. Test groups: card_allowed, minor, wallet_abuse, wallet_recovery, weapon_theme.
  NEVER train on test rows.
- The test set is 10 prompts, so each test prompt moves
  accuracy by ~10 points. Treat test
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

## Investigation 2026-11-15 — why test acc is 0.40 / AUC 0.44 (blocked recall 0.0)

- Labels verified: every one of the 53 prompts was re-run through the live
  `app.guards.check_hard_boundary()` on 2026-11-15 — **0 mismatches**.
  Not a labeling bug. Data is not degenerate either (23 blocked / 30 allowed;
  test 5 blocked / 5 allowed).
- Model config is not the cause: train_acc reaches 0.93, and all 9 experiments
  (different hidden sizes, lr, epochs, seeds) fail identically — blocked
  recall exactly 0.0, fn=5 every time, AUC 0.20–0.44. A config bug would not
  be config-independent.
- Root cause is the grouped split + train-only top-24 word vocab. The test
  blocked rows share almost no vocabulary with train: p005 (wallet_abuse) 0
  in-vocab words, p007 (minor) 0, p049 (wallet_abuse) 0, p004 only "help",
  p006 only "write". Their word-feature vectors are ~all-zero; the 6 shape
  features carry no class signal. Conversely test ALLOWED rows p028/p029 share
  3–4 in-vocab words with BLOCKED train rows ("credit card number stolen",
  "credit card dump"), so the model fires on them (fp 1–3 every run).
  The held-out families ARE the guard's trigger vocabulary — a word-feature
  model cannot generalize to a held-out regex by construction.
- No fix applied: the split is the documented design, and changing the metric
  definition (e.g. random-split primary + grouped as diagnostic, as
  error-inbox does) is a judgment call for Jay, not a clear bug. Options:
  (A) random split primary + grouped diagnostic; (B) keep grouped and accept
  ~0.40 as the honest floor; (C) char/subword features (helps only marginally);
  (D) grow the dataset from real harness logs (none exist yet in the repo).
