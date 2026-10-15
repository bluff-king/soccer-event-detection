"""Text helpers shared by external sources: score/label-leak removal and ASR-style noise."""

from __future__ import annotations

import random
import re

_SCORE_PATTERNS = [
    re.compile(r"\b\d{1,2}\s*[:\-–]\s*\d{1,2}\b"),  # 1:0, 2-1
    re.compile(r"(?:[A-Z][\w.'’-]*\s){1,4}\d{1,2},\s(?:[A-Z0-9][\w.'’-]*\s){1,4}\d{1,2}\.\s*"),  # Everton 1, Leicester City 3.
]
_SPACES = re.compile(r"\s+")


def strip_scores(text: str) -> str:
    for p in _SCORE_PATTERNS:
        text = p.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def asr_normalize(text: str) -> str:
    """Make written text look like Whisper output on commentary: no brackets, light punctuation."""
    text = re.sub(r"[()\[\]\"“”]", " ", text)
    text = re.sub(r"[!;]", ".", text)
    return _SPACES.sub(" ", text).strip()


# Common ASR confusions on football commentary (homophones / near-homophones).
_CONFUSIONS = {
    "goal": ["gold", "go", "goals", "ball"],
    "penalty": ["penalties", "penal tea", "pen alty"],
    "card": ["car", "guard", "cart"],
    "yellow": ["hello", "yell"],
    "scores": ["scorers", "squares"],
    "referee": ["ref", "referees"],
    "keeper": ["keep her", "kipper"],
    "the": ["a", "da"],
}


def asr_noise(text: str, rng: random.Random, p_word: float = 0.05, p_drop_punct: float = 0.7,
              p_truncate: float = 0.2, p_confuse: float = 0.3) -> str:
    """Inject ASR-like noise: drop punctuation, lowercase, homophone confusions, character typos,
    word drops and truncated sentences."""
    if rng.random() < p_drop_punct:
        text = re.sub(r"[.,!?;:]", "", text)
    if rng.random() < 0.5:
        text = text.lower()
    words = text.split()
    out = []
    for w in words:
        lw = w.lower()
        r = rng.random()
        if lw in _CONFUSIONS and rng.random() < p_confuse * 0.3:
            out.append(rng.choice(_CONFUSIONS[lw]))
        elif r < p_word * 0.4:
            continue  # dropped word
        elif r < p_word and len(w) > 3:
            i = rng.randrange(len(w) - 1)
            out.append(w[:i] + w[i + 1] + w[i] + w[i + 2:])  # swapped characters
        else:
            out.append(w)
    if len(out) > 4 and rng.random() < p_truncate:
        cut = rng.randint(max(2, len(out) // 2), len(out) - 1)
        out = out[:cut]
    return " ".join(out) if out else text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in (x.strip() for x in parts) if p]
