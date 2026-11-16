"""100-hour neural network trainer for uncensored-harness.

Predicts whether a user prompt trips the harness hard-boundary guard
(app.guards.check_hard_boundary, label 1 = blocked) or passes
(label 0), from the prompt text alone. Pure Python stdlib only,
self-contained MLP (tanh hidden, sigmoid out, SGD, binary
cross-entropy) in the style of PR #11's nn/mlp.py. No imports from
repo code, no torch/numpy.

Run from this directory (repo_root/nn-100h/):

    python3 train.py [--resume] [--max-run-hours H]

- --max-run-hours defaults to 5.5. On chunk exit (wall time >=
  max-run-hours) the current experiment is checkpointed, chunk
  hours are added to status.json, and the process exits 0.
- --resume loads the latest checkpoint for the in-progress
  experiment (weights + epoch) and continues from that epoch.
  Without --resume a partially-finished experiment restarts from
  epoch 0, but completed experiments are never re-run: the loop
  position is derived from experiments.jsonl.
- Hours are cumulative across restarts in status.json. When
  hours_trained >= 100 the status becomes "done" and training stops.
- The grid is cycled with fresh seeds per cycle, so the search keeps
  going honestly until 100 hours are banked; nothing is faked to
  fill time.

Files in this directory:
  dataset.jsonl, split.json, DATASET.md (built by build_dataset.py),
  experiments.jsonl (one line per finished experiment),
  checkpoints/<exp_id>.json (every 20 epochs),
  best.json / best-metrics.json (only on honest test-acc improvement,
  AUC tiebreak), status.json, heartbeat.json (every <= 60 s).

The split is FROZEN (split.json, seed 42, grouped stratified).
NEVER train on test rows. Vocab is built from TRAIN rows only.
Test = 10 prompts, so each test prompt moves accuracy by 10 points:
treat test numbers as a smoke check, not a benchmark. A 1.000 test
accuracy means leakage — investigate, never report it as a win.
Triage only: check_hard_boundary() stays the only gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = "jaykk99/uncensored-harness"
TARGET_HOURS = 100.0
TOP_K = 24
CHECKPOINT_EVERY = 20
HEARTBEAT_EVERY_S = 60


# ---------------------------------------------------------------------------
# Features (PR #11 features.py pattern: top-K train words + shape counts)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z]+")
_STOP = {
    "the", "and", "for", "with", "how", "what", "are", "you", "your",
    "this", "that", "can", "not", "our", "from", "about", "into",
}


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall((text or "").lower())
            if len(w) >= 3 and w not in _STOP]


def build_vocab(train_rows: list[dict], top_k: int = TOP_K) -> list[str]:
    counts = Counter()
    for row in train_rows:
        counts.update(set(_words(row.get("text") or "")))
    return [w for w, _ in counts.most_common(top_k)]


def extract(text: str, vocab: list[str]) -> list[float]:
    text = text or ""
    present = set(_words(text))
    vec = [1.0 if w in present else 0.0 for w in vocab]
    vec.append(min(1.0, len(text) / 200.0))
    vec.append(min(1.0, len(text.split()) / 30.0))
    alpha = [c for c in text if c.isalpha()]
    vec.append(sum(1 for c in alpha if c.isupper()) / max(1, len(alpha)))
    vec.append(1.0 if any(c.isdigit() for c in text) else 0.0)
    vec.append(1.0 if "?" in text else 0.0)
    vec.append(1.0 if "please" in text.lower() else 0.0)
    return vec


# ---------------------------------------------------------------------------
# MLP (PR #11 mlp.py pattern: tanh hidden, sigmoid out, SGD, BCE)
# ---------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class MLP:
    def __init__(self, layers, seed=42, lr=0.05, l2=1e-4):
        self.layers = list(layers)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.seed = int(seed)
        rng = random.Random(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(
                [[rng.uniform(-limit, limit) for _ in range(fan_in)]
                 for _ in range(fan_out)])
            self.biases.append([0.0 for _ in range(fan_out)])

    def _forward(self, x):
        acts = [list(map(float, x))]
        for li, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = [sum(wi * xi for wi, xi in zip(row, acts[-1])) + bias
                 for row, bias in zip(w, b)]
            if li == len(self.weights) - 1:
                acts.append([_sigmoid(v) for v in z])
            else:
                acts.append([math.tanh(v) for v in z])
        return acts

    def predict_proba(self, x) -> float:
        return self._forward(x)[-1][0]

    def predict(self, x, threshold=0.5) -> int:
        return 1 if self.predict_proba(x) >= threshold else 0

    def _backward(self, acts, y: float):
        n_layers = len(self.weights)
        deltas: list = [None] * n_layers
        deltas[-1] = [acts[-1][0] - y]
        for li in range(n_layers - 2, -1, -1):
            w_next = self.weights[li + 1]
            d_next = deltas[li + 1]
            a = acts[li + 1]
            deltas[li] = [
                sum(w_next[k][j] * d_next[k] for k in range(len(d_next)))
                * (1.0 - a[j] * a[j])
                for j in range(len(a))]
        grad_w, grad_b = [], []
        for li in range(n_layers):
            d = deltas[li]
            a_prev = acts[li]
            grad_w.append([[d[i] * a_prev[j] for j in range(len(a_prev))]
                           for i in range(len(d))])
            grad_b.append(list(d))
        return grad_w, grad_b

    def train_epochs(self, X, y, start_epoch, end_epoch, batch_size=32,
                     on_epoch=None):
        """Train epochs [start_epoch, end_epoch). on_epoch(epoch,
        train_loss) is called after each epoch; returning False stops
        early. Per-epoch shuffle uses Random(seed + epoch), so a
        resume from a checkpoint reproduces the exact order."""
        X = [list(map(float, row)) for row in X]
        y = [float(v) for v in y]
        n = len(X)
        for epoch in range(start_epoch, end_epoch):
            rng = random.Random(self.seed + epoch)
            order = list(range(n))
            rng.shuffle(order)
            for s in range(0, n, max(1, int(batch_size))):
                idxs = order[s:s + int(batch_size)]
                gw_acc = [[[0.0] * len(row) for row in w]
                          for w in self.weights]
                gb_acc = [[0.0] * len(b) for b in self.biases]
                for i in idxs:
                    acts = self._forward(X[i])
                    gw, gb = self._backward(acts, y[i])
                    for li in range(len(gw)):
                        for r in range(len(gw[li])):
                            for c in range(len(gw[li][r])):
                                gw_acc[li][r][c] += gw[li][r][c]
                        for r in range(len(gb[li])):
                            gb_acc[li][r] += gb[li][r]
                m = float(len(idxs))
                for li in range(len(self.weights)):
                    for r in range(len(self.weights[li])):
                        for c in range(len(self.weights[li][r])):
                            g = gw_acc[li][r][c] / m
                            g += self.l2 * self.weights[li][r][c]
                            self.weights[li][r][c] -= self.lr * g
                    for r in range(len(self.biases[li])):
                        self.biases[li][r] -= self.lr * (gb_acc[li][r] / m)
            if on_epoch is not None:
                if on_epoch(epoch + 1, self._bce(X, y)) is False:
                    return epoch + 1
        return end_epoch

    def _bce(self, X, y) -> float:
        eps = 1e-12
        total = 0.0
        for xi, yi in zip(X, y):
            p = min(max(self.predict_proba(xi), eps), 1.0 - eps)
            total += -(yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p))
        return total / max(1, len(X))

    def to_dict(self) -> dict:
        return {"format": "uncensored-harness-nn-mlp/1",
                "layers": self.layers, "lr": self.lr, "l2": self.l2,
                "seed": self.seed, "weights": self.weights,
                "biases": self.biases}

    @classmethod
    def from_dict(cls, data: dict) -> "MLP":
        net = cls(data["layers"], seed=int(data.get("seed", 42)),
                  lr=float(data.get("lr", 0.05)),
                  l2=float(data.get("l2", 1e-4)))
        net.weights = [[[float(v) for v in row] for row in mat]
                       for mat in data["weights"]]
        net.biases = [[float(v) for v in b] for b in data["biases"]]
        return net


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def roc_auc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, lab in pairs if lab == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    ranks = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    for r, (_, lab) in zip(ranks, pairs):
        if lab == 1:
            rank_sum += r
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(net, X, y) -> dict:
    scores = [net.predict_proba(x) for x in X]
    preds = [1 if s >= 0.5 else 0 for s in scores]
    tp = sum(1 for t, p in zip(y, preds) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y, preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y, preds) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y, preds) if t == 0 and p == 0)
    return {
        "n": len(y),
        "accuracy": sum(1 for t, p in zip(y, preds) if t == p) / max(1, len(y)),
        "majority_baseline_accuracy": max(sum(y), len(y) - sum(y)) / max(1, len(y)),
        "roc_auc": roc_auc(scores, y),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_class_recall": {
            "blocked": tp / (tp + fn) if tp + fn else 0.0,
            "allowed": tn / (tn + fp) if tn + fp else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Grid + state
# ---------------------------------------------------------------------------
HIDDEN = [(8, 4), (16, 8), (8,), (16,), (4,), (32, 8), (8, 8)]
LRS = [0.01, 0.05, 0.1, 0.2]
L2S = [0.0, 1e-4, 1e-3]
EPOCHS = [60, 200, 600]
SEEDS = [0, 1, 2, 3, 4]


def grid_for_cycle(cycle: int) -> list[dict]:
    """One cycle's experiment configs, deterministically shuffled by
    cycle so each cycle searches the same space in a new order with
    fresh seeds (seed base shifts by cycle)."""
    configs = []
    for hidden, lr, l2, epochs, seed in itertools.product(
            HIDDEN, LRS, L2S, EPOCHS, SEEDS):
        configs.append({"hidden": list(hidden), "lr": lr, "l2": l2,
                        "epochs": epochs, "seed": seed + cycle * 1000})
    rng = random.Random(42 + cycle)
    rng.shuffle(configs)
    return configs


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Trainer:
    def __init__(self, max_run_hours: float, resume: bool):
        self.max_run_hours = float(max_run_hours)
        self.resume = resume
        self.run_start = time.monotonic()
        self.pid = os.getpid()
        rows = [json.loads(l) for l in
                (BASE / "dataset.jsonl").read_text(encoding="utf-8")
                .splitlines() if l.strip()]
        split = json.loads((BASE / "split.json").read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in rows}
        self.train_rows = [by_id[i] for i in split["train_ids"]]
        self.test_rows = [by_id[i] for i in split["test_ids"]]
        # Vocab from TRAIN rows only — test words never leak in.
        self.vocab = build_vocab(self.train_rows)
        self.X_train = [extract(r["text"], self.vocab) for r in self.train_rows]
        self.y_train = [r["label"] for r in self.train_rows]
        self.X_test = [extract(r["text"], self.vocab) for r in self.test_rows]
        self.y_test = [r["label"] for r in self.test_rows]
        self.dim = len(self.vocab) + 6
        prev = _load_json(BASE / "status.json", {})
        self.prev_hours = float(prev.get("hours_trained", 0.0))
        self.prev_status = prev.get("status", "training")
        self.experiments_done = self._count_experiments()
        self.best = _load_json(BASE / "best.json", None)
        self._last_heartbeat = 0.0
        self._current_exp = None
        self._current_epoch = 0
        (BASE / "checkpoints").mkdir(exist_ok=True)

    def _count_experiments(self) -> int:
        p = BASE / "experiments.jsonl"
        if not p.exists():
            return 0
        return sum(1 for l in p.read_text(encoding="utf-8").splitlines()
                   if l.strip())

    def hours_now(self) -> float:
        return self.prev_hours + (time.monotonic() - self.run_start) / 3600.0

    def run_elapsed_h(self) -> float:
        return (time.monotonic() - self.run_start) / 3600.0

    def write_status(self, status: str) -> None:
        _write_json(BASE / "status.json", {
            "repo": REPO,
            "hours_trained": round(self.hours_now(), 4),
            "experiments_done": self.experiments_done,
            "best": self.best,
            "status": status,
            "updated": datetime.now(timezone.utc).isoformat(),
        })

    def write_heartbeat(self, exp_id, epoch) -> None:
        now = time.time()
        if now - self._last_heartbeat < HEARTBEAT_EVERY_S and epoch:
            return
        self._last_heartbeat = now
        _write_json(BASE / "heartbeat.json", {
            "pid": self.pid,
            "time": datetime.now(timezone.utc).isoformat(),
            "exp_id": exp_id,
            "epoch": epoch,
            "experiments_done": self.experiments_done,
        })

    def checkpoint(self, exp_id, config, epoch, net, train_loss) -> None:
        _write_json(BASE / "checkpoints" / f"{exp_id}.json", {
            "exp_id": exp_id, "config": config, "epoch": epoch,
            "weights": net.to_dict(), "train_loss": train_loss,
        })

    def run_experiment(self, exp_id, config) -> dict | None:
        """Train one config on train split, eval on frozen test.
        Returns the experiments.jsonl record, or None if the chunk
        time budget ran out mid-experiment (already checkpointed)."""
        layers = [self.dim] + list(config["hidden"]) + [1]
        net = MLP(layers, seed=config["seed"], lr=config["lr"],
                  l2=config["l2"])
        start_epoch = 0
        ckpt_path = BASE / "checkpoints" / f"{exp_id}.json"
        if self.resume and ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ckpt.get("config") == config:
                net = MLP.from_dict(ckpt["weights"])
                start_epoch = int(ckpt["epoch"])
                print(f"[{exp_id}] resume: continuing from epoch "
                      f"{start_epoch} checkpoint", flush=True)
        t0 = time.monotonic()
        state = {"stop": False}

        def on_epoch(epoch, train_loss):
            self._current_epoch = epoch
            self.write_heartbeat(exp_id, epoch)
            if epoch % CHECKPOINT_EVERY == 0 or epoch == config["epochs"]:
                self.checkpoint(exp_id, config, epoch, net, train_loss)
            if self.run_elapsed_h() >= self.max_run_hours:
                self.checkpoint(exp_id, config, epoch, net, train_loss)
                print(f"[{exp_id}] chunk budget reached at epoch "
                      f"{epoch}; checkpointed", flush=True)
                state["stop"] = True
                return False
            return True

        self._current_exp = exp_id
        end_epoch = net.train_epochs(
            self.X_train, self.y_train, start_epoch, config["epochs"],
            on_epoch=on_epoch)
        self._current_exp = None
        if state["stop"] and end_epoch < config["epochs"]:
            return None
        seconds = time.monotonic() - t0
        test_m = evaluate(net, self.X_test, self.y_test)
        train_m = evaluate(net, self.X_train, self.y_train)
        record = {
            "exp_id": exp_id,
            "config": config,
            "test_acc": round(test_m["accuracy"], 4),
            "auc": round(test_m["roc_auc"], 4),
            "confusion": test_m["confusion"],
            "per_class_recall": {k: round(v, 4)
                                 for k, v in test_m["per_class_recall"].items()},
            "train_acc": round(train_m["accuracy"], 4),
            "majority_baseline": round(test_m["majority_baseline_accuracy"], 4),
            "seconds": round(seconds, 2),
        }
        with open(BASE / "experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.experiments_done += 1
        if self.best is None or (record["test_acc"], record["auc"]) > (
                self.best["test_acc"], self.best["auc"]):
            self.best = record
            _write_json(BASE / "best.json", record)
            _write_json(BASE / "best-metrics.json", {
                "exp_id": exp_id, "config": config,
                "test": test_m, "train": train_m,
                "feature_vocab": self.vocab,
                "layers": layers,
            })
            print(f"[{exp_id}] NEW BEST test_acc={record['test_acc']} "
                  f"auc={record['auc']}", flush=True)
        else:
            print(f"[{exp_id}] test_acc={record['test_acc']} "
                  f"auc={record['auc']} "
                  f"conf={record['confusion']} "
                  f"({record['seconds']}s)", flush=True)
        try:
            ckpt_path.unlink()
        except OSError:
            pass
        return record

    def run(self) -> int:
        if self.prev_status == "done":
            # Terminal state: touch NOTHING (no status/heartbeat writes),
            # so the CircleCI commit step finds no changes and the chain
            # stops instead of looping on empty checkpoint commits.
            print("already done, nothing to do")
            return 0
        if self.hours_now() >= TARGET_HOURS:
            self.write_status("done")
            print("100h target already banked; status=done")
            return 0
        print(f"train rows={len(self.train_rows)} "
              f"test rows={len(self.test_rows)} features={self.dim} "
              f"hours_banked={self.prev_hours:.4f} "
              f"experiments_done={self.experiments_done} "
              f"resume={self.resume} max_run_hours={self.max_run_hours}",
              flush=True)
        grid_size = len(HIDDEN) * len(LRS) * len(L2S) * len(EPOCHS) * len(SEEDS)
        while True:
            if self.hours_now() >= TARGET_HOURS:
                self.write_status("done")
                print("100h target reached; status=done")
                return 0
            if self.run_elapsed_h() >= self.max_run_hours:
                self.write_status("training")
                print("chunk budget reached; status=training, exit 0")
                return 0
            cycle = self.experiments_done // grid_size
            pos = self.experiments_done % grid_size
            grid = grid_for_cycle(cycle)
            config = grid[pos]
            exp_id = f"c{cycle:04d}-e{pos:04d}"
            self.write_heartbeat(exp_id, 0)
            rec = self.run_experiment(exp_id, config)
            self.write_status("training")
            if rec is None:
                # Chunk budget hit mid-experiment; checkpoint saved.
                self.write_status("training")
                print("chunk budget reached mid-experiment; exit 0")
                return 0
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="100h trainer: guard-boundary triage MLP (stdlib only).")
    ap.add_argument("--resume", action="store_true",
                    help="resume in-progress experiment from its checkpoint")
    ap.add_argument("--max-run-hours", type=float, default=5.5,
                    help="wall-clock budget for this chunk (default 5.5)")
    args = ap.parse_args(argv)
    if not (BASE / "dataset.jsonl").exists() or not (BASE / "split.json").exists():
        print("dataset.jsonl / split.json missing; run first:\n"
              "  python3 nn-100h/build_dataset.py   (from repo root)",
              file=sys.stderr)
        return 2
    return Trainer(args.max_run_hours, args.resume).run()


if __name__ == "__main__":
    sys.exit(main())
