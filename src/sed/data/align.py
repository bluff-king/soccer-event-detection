"""Delay-aware alignment between ASR segments and ground-truth events.

Commentators talk about an event a few seconds *after* it happens. A segment is labelled with class
``c`` if it overlaps the interval ``[t + delay_min, t + delay_max]`` of an event of class ``c`` at
time ``t`` (same half). ``delay_min``/``delay_max`` are hyperparameters; ``measure_delays`` estimates
their empirical distribution from keyword mentions so they can be chosen from data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..labels import DEFAULT_PRIORITY, NO_EVENT, CLASSES
from .types import Event, Segment

# Words a commentator typically uses when the event actually happens (English + a few
# Spanish/French/Italian/German stems because some untranslated text remains).
CLASS_KEYWORDS = {
    "Goal": r"\b(goa+l+|gol+|scores?|scored|scoring|net|tor|but)\b",
    "Card": r"\b(card|yellow|red|booked|booking|book|sent off|caution|cautioned|tarjeta|carton)\b",
    "Penalty": r"\b(penalt(y|ies)|spot[- ]kick|spot|penal|rigore|elfmeter)\b",
}
_KW = {k: re.compile(v, re.IGNORECASE) for k, v in CLASS_KEYWORDS.items()}


def mentions(cls: str, text: str) -> bool:
    return bool(_KW[cls].search(text))


def mentions_any(text: str) -> list[str]:
    return [c for c in _KW if _KW[c].search(text)]


def label_segments(
    segments: list[Segment],
    events: list[Event],
    half: int,
    delay_min: float = 0.0,
    delay_max: float = 20.0,
    priority: list[str] | None = None,
    per_class: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Return one task label per segment (``"No-Event"`` when no event interval overlaps).

    ``per_class`` optionally overrides ``(delay_min, delay_max)`` per class, e.g. Labels-v2 marks a
    Penalty when the kick is taken while commentators discuss it from the moment it is awarded,
    i.e. 30-60 s *before* the label (see docs/DATA_SURVEY.md)."""
    priority = priority or DEFAULT_PRIORITY
    rank = {c: i for i, c in enumerate(priority)}
    evs = [e for e in events if e.half == half and e.cls]
    labels = [CLASSES[NO_EVENT]] * len(segments)
    for i, s in enumerate(segments):
        best = None
        for e in evs:
            dmin, dmax = (per_class or {}).get(e.cls, (delay_min, delay_max))
            lo, hi = e.time + dmin, e.time + dmax
            if s.start < hi and s.end > lo:  # interval overlap
                if best is None or rank.get(e.cls, 99) < rank.get(best, 99):
                    best = e.cls
        if best:
            labels[i] = best
    return labels


def event_segment_index(segments: list[Segment], events: list[Event], half: int,
                        delay_min: float, delay_max: float,
                        per_class: dict[str, tuple[float, float]] | None = None) -> list[tuple[Event, list[int]]]:
    """For each highlight event in this half, the indices of segments inside its label interval."""
    out = []
    for e in events:
        if e.half != half or not e.cls:
            continue
        dmin, dmax = (per_class or {}).get(e.cls, (delay_min, delay_max))
        lo, hi = e.time + dmin, e.time + dmax
        idx = [i for i, s in enumerate(segments) if s.start < hi and s.end > lo]
        out.append((e, idx))
    return out


def mention_profile(segments: list[Segment], events: list[Event], half: int, cls: str,
                    lo: float = -60.0, hi: float = 120.0, bin_s: float = 5.0) -> list[int]:
    """Histogram (counts per ``bin_s`` bin) of keyword mentions of ``cls`` relative to the events of
    that class. The excess over the flat baseline shows *when* commentators talk about an event."""
    nb = int((hi - lo) / bin_s)
    hist = [0] * nb
    for e in events:
        if e.half != half or e.cls != cls:
            continue
        for s in segments:
            off = s.start - e.time
            if lo <= off < hi and mentions(cls, s.text):
                hist[int((off - lo) // bin_s)] += 1
    return hist


@dataclass
class DelayObs:
    game: str
    half: int
    cls: str
    event_time: float
    delay: float | None  # first keyword mention (segment start) minus event time; None = no mention
    text: str = ""


def measure_delays(
    segments: list[Segment],
    events: list[Event],
    half: int,
    game: str = "",
    search_before: float = 30.0,
    search_after: float = 90.0,
) -> list[DelayObs]:
    """Commentator reaction delay per event: time of the first segment mentioning the event's
    class keyword within ``[t - search_before, t + search_after]``."""
    out = []
    for e in events:
        if e.half != half or not e.cls:
            continue
        hit = None
        for s in segments:
            if s.end < e.time - search_before:
                continue
            if s.start > e.time + search_after:
                break
            if mentions(e.cls, s.text):
                hit = s
                break
        out.append(DelayObs(game, half, e.cls, e.time,
                            None if hit is None else hit.start - e.time,
                            "" if hit is None else hit.text))
    return out
