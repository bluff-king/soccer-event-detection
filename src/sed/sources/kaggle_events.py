"""Kaggle "Football Events" (secareanualin/football-events): ~941k templated live-text events from
9,074 games (top-5 European leagues, 2011/12-2016/17).

The text is highly templated ("Goal! Everton 1, Leicester City 3. ...", "X is shown the yellow card"),
so a model could learn the template instead of the semantics. We therefore strip the class-specific
template markers, explicit label keywords and scores before use, and only use it for training
(pre-training / supplementing), never for evaluation.

Games that are also SoccerNet games in valid/test (same date and teams) are removed.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from pathlib import Path

from ..data.types import Sample
from .text_utils import asr_normalize, strip_scores

# Template markers / label keywords to remove (applied in order, case-insensitive).
_LEAKS = [
    r"^\s*goal!\s*(goal!)?",
    r"^\s*booking\b",
    r"^\s*dismissal\b",
    r"^\s*attempt (missed|blocked|saved)\.",
    r"^\s*corner,\s*",
    r"^\s*offside,\s*",
    r"^\s*substitution,\s*",
    r"^\s*penalty conceded by\b",
    r"^\s*penalty\s+[^.]+\.",  # "Penalty Napoli."
    r"^\s*second yellow card to\b",
    r"^\s*hand ball by\b",
    r"^\s*foul by\b",
    r"\bis shown the (yellow|red) card\b",
    r"\b(yellow|red) card\b",
    r"\bdraws a foul in the penalty area\b",
    r"\bafter a foul in the penalty area\b",
    r"\b(converts|misses|missed|saves|saved) the penalty\b",
    r"\bpenalty\b",
    r"\bown goal\b",
]
_LEAK_RE = [re.compile(p, re.IGNORECASE) for p in _LEAKS]


def mask_leaks(text: str) -> str:
    text = strip_scores(text)
    for r in _LEAK_RE:
        text = r.sub(" ", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    return asr_normalize(text).strip(" .,")


def kaggle_label(row: dict) -> str:
    try:
        is_goal = int(row.get("is_goal") or 0)
        et = int(float(row.get("event_type") or -1))
    except ValueError:
        return "No-Event"
    if is_goal == 1:
        return "Goal"
    if et in (4, 5, 6):
        return "Card"
    if et == 11:
        return "Penalty"
    return "No-Event"


def _norm_team(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\b(fc|cf|ac|as|ss|sc|afc|calcio|club|de|real|united|city|1\.)\b", " ", name)
    return re.sub(r"[^a-z]", "", name)[:6]


def soccernet_game_key(game: str) -> tuple[date, str, str] | None:
    """'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley' -> (date, home, away)."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2}) - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$", game.split("/")[-1])
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), _norm_team(m.group(4)), _norm_team(m.group(5))


def overlapping_kaggle_games(ginf_rows: list[dict], soccernet_games: list[str]) -> set[str]:
    """id_odsp of Kaggle games that coincide (date +-1 day, both teams) with given SoccerNet games."""
    keys = [k for k in (soccernet_game_key(g) for g in soccernet_games) if k]
    idx: dict[date, list[tuple[str, str]]] = {}
    for d, h, a in keys:
        idx.setdefault(d, []).append((h, a))
    out = set()
    for r in ginf_rows:
        try:
            d = date.fromisoformat(r["date"])
        except (KeyError, ValueError):
            continue
        h, a = _norm_team(r.get("ht", "")), _norm_team(r.get("at", ""))
        for dd in (d - timedelta(days=1), d, d + timedelta(days=1)):
            for sh, sa in idx.get(dd, []):
                if (sh and sa) and ({sh, sa} == {h, a} or (sh[:4] == h[:4] and sa[:4] == a[:4])):
                    out.add(r["id_odsp"])
    return out


def load_kaggle_samples(root: str | Path, exclude_games: set[str] | None = None, max_per_class: int | None = None,
                        neg_per_pos: float = 1.0, seed: int = 42, mask: bool = True) -> list[Sample]:
    import csv

    root = Path(root)
    exclude_games = exclude_games or set()
    rng = random.Random(seed)
    by_cls: dict[str, list[Sample]] = {"Goal": [], "Card": [], "Penalty": [], "No-Event": []}
    with (root / "events.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["id_odsp"] in exclude_games:
                continue
            lab = kaggle_label(row)
            text = mask_leaks(row["text"]) if mask else row["text"]
            if len(text.split()) < 2:
                continue
            by_cls[lab].append(Sample(text=text, label=lab, source="kaggle", game="kaggle:" + row["id_odsp"],
                                      split="train", meta={"event_type": row["event_type"]}))
    out = []
    n_pos = 0
    for c in ("Goal", "Card", "Penalty"):
        items = by_cls[c]
        rng.shuffle(items)
        if max_per_class:
            items = items[:max_per_class]
        n_pos += len(items)
        out += items
    negs = by_cls["No-Event"]
    rng.shuffle(negs)
    out += negs[: int(n_pos * neg_per_pos)]
    return out


def kaggle_class_counts(root: str | Path) -> dict[str, int]:
    import csv

    counts = {"Goal": 0, "Card": 0, "Penalty": 0, "No-Event": 0}
    with (Path(root) / "events.csv").open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            counts[kaggle_label(row)] += 1
    return counts
