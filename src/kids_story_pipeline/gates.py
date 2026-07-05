"""Deterministic kid/parent gate logic (real code, not LLM).

These run in BOTH mock and real mode. In real mode an LLM persona review is
layered on top; these hard checks are the floor that never gets skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z']+")


def _syllables(word: str) -> int:
    groups = _VOWEL_GROUP.findall(word)
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch_kincaid_grade(text: str) -> float:
    sentences = [s for s in _SENT_SPLIT.split(text) if _WORD.search(s)]
    words = _WORD.findall(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return round(max(0.0, grade), 2)


@dataclass
class GateResult:
    name: str
    confidence: float          # 0..1, higher = safer/better
    passed: bool
    details: str


def reading_level_gate(text: str, max_grade: float) -> GateResult:
    grade = flesch_kincaid_grade(text)
    if grade <= max_grade:
        conf = 1.0
    else:
        # degrade 0.15 per grade above ceiling
        conf = max(0.0, 1.0 - 0.15 * (grade - max_grade))
    return GateResult(
        name="reading_level",
        confidence=round(conf, 3),
        passed=grade <= max_grade + 2.0,   # hard fail only if wildly above
        details=f"FK grade {grade} vs ceiling {max_grade}",
    )


def scary_content_gate(text: str, scary_words: list[str]) -> GateResult:
    lowered = text.lower()
    hits = sorted({w for w in scary_words if re.search(rf"\b{re.escape(w.lower())}\b", lowered)})
    conf = max(0.0, 1.0 - 0.25 * len(hits))
    return GateResult(
        name="scary_content",
        confidence=round(conf, 3),
        passed=len(hits) == 0,
        details=("no flagged terms" if not hits else f"flagged: {', '.join(hits)}"),
    )


def hook_gate(first_lines: str, max_words: int = 45) -> GateResult:
    """Ages 4-8 bedtime: opening should reach the inciting image fast."""
    words = len(_WORD.findall(first_lines))
    conf = 1.0 if words <= max_words else max(0.3, 1.0 - 0.02 * (words - max_words))
    return GateResult(
        name="hook",
        confidence=round(conf, 3),
        passed=True,
        details=f"{words} words before first scene turn (target <= {max_words})",
    )


def combine(results: list[GateResult]) -> float:
    """Overall confidence = min of components (weakest link)."""
    return round(min(r.confidence for r in results), 3) if results else 1.0
