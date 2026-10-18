"""Narrow hard-boundary pre-check. Only the truly hard lines live here.

Everything else passes straight to the model. This is a surgical guard,
not a content filter — legit work always passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryHit:
    reason: str
    alternative: str


# (pattern, reason, alternative) — keep this list tiny and only for hard lines.
_BOUNDARIES = [
    
]


def check_hard_boundary(text: str) -> BoundaryHit | None:
    t = (text or "").lower()
    for pat, reason, alt in _BOUNDARIES:
        if re.search(pat, t):
            return BoundaryHit(reason=reason, alternative=alt)
    return None
