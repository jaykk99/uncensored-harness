"""Feature extraction: user prompt text -> fixed-length vector.

Every feature comes from the prompt text itself: which of the
top-K words learned from training prompts it contains, plus a few
length/shape counts. Labels never leak into features: whether the
guard blocked the prompt is label-side only and is never read here.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

TOP_K = 24

_WORD_RE = re.compile(r"[a-z]+")
_STOP = {
    "the", "and", "for", "with", "how", "what", "are", "you", "your",
    "this", "that", "can", "not", "our", "from", "about", "into",
}


def _words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall((text or "").lower())
            if len(w) >= 3 and w not in _STOP]


class FeatureSpec:
    """Top-K word vocab + layout for the feature vector."""

    def __init__(self, words: Sequence[str] = ()) -> None:
        self.words = list(words)

    @property
    def dim(self) -> int:
        return len(self.feature_names())

    def feature_names(self) -> List[str]:
        names: List[str] = []
        names += [f"word:{w}" for w in self.words]
        names += ["len", "nwords", "upper_ratio", "has_digit",
                  "has_question", "has_please"]
        return names

    def groups(self) -> Dict[str, Tuple[int, int]]:
        names = self.feature_names()
        idx = {n: i for i, n in enumerate(names)}

        def span(prefix: str) -> Tuple[int, int]:
            hits = [i for n, i in idx.items() if n.startswith(prefix)]
            return (min(hits), max(hits) + 1)

        return {
            "words": span("word:"),
            "shape": (idx["len"], idx["has_please"] + 1),
        }

    def to_dict(self) -> Dict:
        return {"words": self.words}

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureSpec":
        return cls(data.get("words", []))


def build_vocab(rows: Sequence[Dict], top_k: int = TOP_K) -> FeatureSpec:
    """Learn the top-K words from training prompts."""
    from collections import Counter
    counts = Counter()
    for row in rows:
        counts.update(set(_words(row.get("text") or "")))
    return FeatureSpec([w for w, _ in counts.most_common(top_k)])


def extract(text: str, spec: FeatureSpec) -> List[float]:
    """One prompt -> feature vector in spec order."""
    text = text or ""
    present = set(_words(text))
    vec: List[float] = []
    vec += [1.0 if w in present else 0.0 for w in spec.words]
    vec.append(min(1.0, len(text) / 200.0))
    vec.append(min(1.0, len(text.split()) / 30.0))
    alpha = [c for c in text if c.isalpha()]
    vec.append(sum(1 for c in alpha if c.isupper()) / max(1, len(alpha)))
    vec.append(1.0 if any(c.isdigit() for c in text) else 0.0)
    vec.append(1.0 if "?" in text else 0.0)
    vec.append(1.0 if "please" in text.lower() else 0.0)
    return vec


def top_features(text: str, spec: FeatureSpec, net,
                 k: int = 3) -> List[Dict]:
    """Top contributing features for one scored prompt (leave-one-out)."""
    x = extract(text, spec)
    base = net.predict_proba(x)
    names = spec.feature_names()
    deltas = []
    for i, name in enumerate(names):
        if x[i] == 0.0:
            continue
        x2 = list(x)
        x2[i] = 0.0
        d = base - net.predict_proba(x2)
        deltas.append((abs(d), d, name))
    deltas.sort(reverse=True)
    return [{"feature": name,
             "direction": "up" if d > 0 else "down",
             "delta": round(d, 4)}
            for _abs, d, name in deltas[:k]]
