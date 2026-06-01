from __future__ import annotations

import re
from collections import Counter
from typing import Any


ANSWER_MARKER_RE = re.compile(r"####|final answer:|answer:|final:", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

PENALTIES = {
    "empty_candidate": 1.0,
    "tiny_candidate": 0.5,
    "missing_answer_marker": 0.15,
    "repetition_collapse": 0.35,
    "ngram_collapse": 0.35,
    "too_long": 0.15,
}


def _trigram_counts(words: list[str]) -> Counter[tuple[str, str, str]]:
    return Counter(zip(words, words[1:], words[2:]))


def evaluate_soft_checks(candidate: str) -> dict[str, Any]:
    stripped = candidate.strip()
    has_answer_marker = ANSWER_MARKER_RE.search(candidate) is not None
    words = [match.group(0).lower() for match in WORD_RE.finditer(candidate)]
    word_counts = Counter(words)
    max_word_count = max(word_counts.values(), default=0)
    max_word_fraction = max_word_count / len(words) if words else 0.0
    trigram_counts = _trigram_counts(words)
    max_trigram_count = max(trigram_counts.values(), default=0)

    flags: list[str] = []
    if not stripped:
        flags.append("empty_candidate")
    elif len(stripped) < 3:
        flags.append("tiny_candidate")
    if not has_answer_marker:
        flags.append("missing_answer_marker")
    if len(words) >= 8 and max_word_fraction >= 0.45:
        flags.append("repetition_collapse")
    if max_trigram_count >= 3:
        flags.append("ngram_collapse")
    if len(candidate) > 500:
        flags.append("too_long")

    penalty = sum(PENALTIES[flag] for flag in flags)
    score = max(0.0, min(1.0, round(1.0 - penalty, 10)))
    return {
        "score": score,
        "flags": flags,
        "metrics": {
            "char_count": len(candidate),
            "word_count": len(words),
            "has_answer_marker": has_answer_marker,
            "max_word_count": max_word_count,
            "max_word_fraction": max_word_fraction,
            "max_trigram_count": max_trigram_count,
        },
    }
