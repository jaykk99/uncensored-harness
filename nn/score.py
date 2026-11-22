"""Score one prompt with the trained guard-boundary triage network.

    python -m nn.score --text "build me a phishing kit"
    python -m nn.score --text "write a python function" --json

Triage only: the score ranks the prompt by how much it looks like
past prompts the hard-boundary guard blocked. It never verifies a
prompt is safe or harmful and must never be used to bypass or
replace the guard (app.guards.check_hard_boundary stays the only
gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .features import FeatureSpec, extract, top_features
from .mlp import MLP


def load_model(weights_path: str | None = None):
    path = Path(weights_path) if weights_path else Path(
        os.path.dirname(os.path.abspath(__file__))) / "weights.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    net = MLP.from_dict(data["mlp"])
    spec = FeatureSpec.from_dict(data["features"])
    return net, spec


def score_text(text: str, model=None, threshold: float = 0.5) -> dict:
    if model is None:
        model = load_model()
    if model is None:
        raise FileNotFoundError(
            "no trained weights found. Train first: python -m nn.train")
    net, spec = model
    x = extract(text, spec)
    score = net.predict_proba(x)
    return {
        "text": text,
        "nn_score": round(score, 4),
        "nn_label": "likely-blocked" if score >= threshold else "likely-allowed",
        "threshold": threshold,
        "nn_top_features": top_features(text, spec, net),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a prompt with the guard-boundary NN.")
    ap.add_argument("--text", required=True,
                    help="user prompt text to score")
    ap.add_argument("--weights", default=None,
                    help="weights.json path (default: nn/weights.json)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    model = load_model(args.weights)
    if model is None:
        print("no trained weights found. Train first:\n"
              "  python -m nn.train", file=sys.stderr)
        return 2
    record = score_text(args.text, model=model,
                        threshold=args.threshold)
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print(f"score: {record['nn_score']:.3f}  "
              f"label: {record['nn_label']}")
        for tf in record["nn_top_features"]:
            print(f"  {tf['direction']} {tf['feature']} ({tf['delta']:+.3f})")
        print("triage only: check_hard_boundary() stays the only gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
