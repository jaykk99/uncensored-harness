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
    (r"credit card (number|dump|steal)|stolen card|cvv.*\d{3}",
     "Can't help with stolen-card stuff.",
     "If it's your own card or a test environment, tell me and I'll help with that instead."),
    (r"phishing (kit|page|email)|credential harvest",
     "Can't build phishing kits.",
     "I can help with phishing *defense* — detection rules, training material, email filters."),
    (r"drain.*wallet|steal.*(crypto|bitcoin|funds)",
     "Can't help drain or steal anyone's funds.",
     "If it's your own wallet recovery, describe the situation and I'll help."),
    (r"(child|minor).*(sexual|porn)|csam",
     "Can't help with anything sexual involving minors, ever.",
     "Nothing to offer here."),
    (r"(bioweapon|nuke|dirty bomb).*(build|make|instructions)",
     "Can't help with weapons of mass harm.",
     "Happy to help with anything else technical."),
]


def check_hard_boundary(text: str) -> BoundaryHit | None:
    t = (text or "").lower()
    for pat, reason, alt in _BOUNDARIES:
        if re.search(pat, t):
            return BoundaryHit(reason=reason, alternative=alt)
    return None
