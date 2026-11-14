# nn-100h — 100-hour guard-triage trainer (uncensored-harness)

Trains a small pure-stdlib MLP to predict whether a user prompt trips
the harness hard-boundary guard (`app.guards.check_hard_boundary`,
label 1 = blocked) or passes (label 0), from the prompt text alone.

## Layout

- `build_dataset.py` — builds `dataset.jsonl`, `split.json`,
  `DATASET.md` from real repo prompts only (see DATASET.md for
  provenance). Run once from repo root:
  `python3 nn-100h/build_dataset.py`
- `train.py` — the trainer. Self-contained, stdlib only, no imports
  from repo code. Run from this directory:
  `python3 train.py [--resume] [--max-run-hours H]`
- `dataset.jsonl` — 53 real prompts, labels from running the actual
  guard (23 blocked / 30 allowed).
- `split.json` — frozen grouped stratified split, seed 42, test 10
  rows (whole topic families held out; never train on test rows).
- `experiments.jsonl` — one line per finished experiment.
- `checkpoints/<exp_id>.json` — every 20 epochs.
- `best.json` / `best-metrics.json` — best honest test result
  (test-acc improvement, AUC tiebreak).
- `status.json` — cumulative hours, experiments done,
  `training` | `done` (done at 100 h).
- `heartbeat.json` — updated at least every 60 s while training.

## Running

Local (this VM reboots ~every 2 h; the trainer is built for that):

```sh
cd nn-100h
python3 train.py --max-run-hours 5.5   # one chunk, exits 0 on budget
python3 train.py --resume --max-run-hours 5.5   # continue after a kill
```

GitHub Actions: `.github/workflows/nn-100h.yml` (manual dispatch,
`max_run_hours` input) runs the same command and uploads
`experiments.jsonl`, `best.json`, `status.json` as artifacts.

## Honest limits

- The labeled set is TINY: 53 prompts, test set 10. Each test prompt
  moves accuracy by 10 points. Treat test numbers as a smoke check.
- The guard is a deterministic rule, so this is rule reproduction:
  the grouped split holds out whole prompt families (minor,
  wallet_abuse, wallet_recovery, card_allowed, weapon_theme) to
  measure generalization to unseen keywords. A 1.000 test accuracy
  means leakage — investigate, never report it as a win.
- Triage only. `check_hard_boundary()` stays the only gate; a low
  score never means a prompt is safe. Never use this to bypass,
  weaken, or replace the guard.
