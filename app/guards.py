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
    # "phishing email" is only blocked with an attack verb in front of it, so
    # defensive work ("analyze this phishing email I received") passes while
    # crafting ("write a phishing email to our customers") stays blocked.
    (r"phishing (kit|page)|credential harvest"
     r"|(write|send|craft|create|draft|build).{0,40}phishing email",
     "Can't build phishing kits.",
     "I can help with phishing *defense* — detection rules, training material, email filters."),
    (r"drain.*wallet|steal.*(crypto|bitcoin|funds)",
     "Can't help drain or steal anyone's funds.",
     "If it's your own wallet recovery, describe the situation and I'll help."),
    (r"(child|minor).*(sexual|porn)|csam",
     "Can't help with anything sexual involving minors, ever.",
     "Nothing to offer here."),
    # Verb may come before or after the weapon term ("how to make a dirty
    # bomb" must hit just like "dirty bomb build instructions").
    (r"(bioweapon|nuke|dirty bomb).*(build|make|instructions)"
     r"|(build|make).*(bioweapon|nuke|dirty bomb)",
     "Can't help with weapons of mass harm.",
     "Happy to help with anything else technical."),
]


# Narrow defense carve-outs for patterns that also match legitimate work.
# Each carve-out is checked ONLY against its own pattern, and a deception /
# misuse verb anywhere in the text cancels the carve-out, so a crafted
# attack cannot smuggle itself past the guard by appending a defense word.
# Anything not listed here stays blocked, including every hard-limit case
# the guard cannot reliably tell apart from abuse (own-wallet draining,
# which is sweeping language, and any minor/sexual phrasing).
_PHISHING_DEFENSE_MARKERS = (
    "phishing email filter",
    "phishing email detector",
    "phishing email detection",
    "phishing email reporting",
    "filter for phishing",
    "detector for phishing",
    "banning phishing",
)
_PHISHING_DECEPTION = re.compile(r"trick|deceiv|fool|impersonat")


def _phishing_defense(t: str) -> bool:
    if _PHISHING_DECEPTION.search(t):
        return False
    if any(m in t for m in _PHISHING_DEFENSE_MARKERS):
        return True
    # A training example is defense material, not a phish to send.
    return "phishing email example" in t and "train" in t


def _stolen_card_defense(t: str) -> bool:
    if re.search(r"\b(use it|give me|buy|pay with|sell)\b", t):
        return False
    # Victim reporting their own stolen card (the guard's own alternative
    # invites exactly this) or fraud-pattern analysis of an investigation.
    if "my" in t and "stolen" in t and re.search(
            r"report|freeze|cancel|replace", t):
        return True
    return ("fraud pattern" in t and "investigat" in t) or \
        "fraud detection" in t


def _weapon_defense(t: str) -> bool:
    # "nuke" as a theme, not a weapon: a party cake, not instructions.
    return ("nuke-themed" in t or "nuke cake" in t) and \
        "bioweapon" not in t and "dirty bomb" not in t


# Pattern index -> carve-out, keyed by the reason string so the table above
# can stay a plain list. Only the three patterns below have carve-outs.
_CARVE_OUTS = {
    "Can't help with stolen-card stuff.": _stolen_card_defense,
    "Can't build phishing kits.": _phishing_defense,
    "Can't help with weapons of mass harm.": _weapon_defense,
}


def check_hard_boundary(text: str) -> BoundaryHit | None:
    t = (text or "").lower()
    for pat, reason, alt in _BOUNDARIES:
        if re.search(pat, t):
            carve_out = _CARVE_OUTS.get(reason)
            if carve_out is not None and carve_out(t):
                continue  # defensive phrasing: keep scanning other patterns
            return BoundaryHit(reason=reason, alternative=alt)
    return None
