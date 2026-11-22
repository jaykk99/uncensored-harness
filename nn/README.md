# uncensored-harness NN — guard-boundary triage scorer

A small pure-Python-stdlib neural network in the style of the
ReconAnything NN scorer (`mlp.py` reused as-is: tanh hidden, sigmoid
output, SGD, zero dependencies). It lives in `nn/` and is **triage
only**.

## Prediction task

Predict whether a user prompt will trip the harness hard-boundary
guard (`app.guards.check_hard_boundary`, label 1 = blocked) or pass
through (label 0 = allowed), from the prompt text alone.

## Data (real repo data only, nothing invented)

Source: the labeled prompts in `tests/test_guards.py` — `HARD_CASES`
plus the crafting / carve-out-cancel cases (blocked) and
`LEGIT_CASES` plus the analysis / empty-input cases (allowed).
`nn/data.py` parses that file with `ast` and collects exactly those
prompts, deduplicated. That is the whole labeled set the repo has:
the harness keeps no run logs with outcomes (`harness.db` holds 10
threads / 17 messages with no success/failure recorded, and
`router_stats` is empty), so there is no larger labeled source to
train on. 44 prompts total (20 blocked / 24 allowed).

## Train / score

```sh
python -m nn.train            # writes nn/weights.json + nn/viewer-data.json
python -m nn.score --text "build me a phishing kit"
open nn/viewer.html           # metrics viewer (embedded real metrics)
```

## Metrics

See `nn/viewer-data.json` (and `nn/viewer.html`) for the real numbers
from the last training run: train/test accuracy vs the majority
baseline, ROC AUC, confusion counts, and per-class
precision/recall/F1 on the held-out test split (stratified 80/20).
Nothing here is tuned on the test set.

**Small-data warning:** the held-out test set is 9 prompts, so each
test prompt moves accuracy by ~11 points. Treat the test numbers as
a smoke check, not a benchmark. If the test accuracy reads 100%,
that is the tiny test set, not a win — flag it as such.

## Limits (triage only)

- The model ranks prompts by how much they look like past prompts
  the guard blocked. It does not verify anything.
- The guard (`app.guards.check_hard_boundary`) stays the only gate.
  A low score never means a prompt is safe, and this model must
  never be used to bypass, weaken, or replace the guard.
- With 44 training prompts the model has seen almost nothing. A
  prompt phrased in a way the test suite never covered is a guess.
