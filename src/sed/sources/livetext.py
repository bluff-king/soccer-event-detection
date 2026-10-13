"""Real live-text commentary: Hugging Face ``falahmanalodi/Football-Commentary`` (MIT). 189 Premier League 2024-25
matches, ~25k minute-stamped lines with an event type per line (G, OG, PSG = penalty scored, PM = penalty missed,
"penalty save", YC/RC/Y2C, var, highlight, comment, ...). The matches are years after the SoccerNet games
(2014-2020), so no valid/test game can leak in. Train-only, like every external source.

Labels follow the delay-window logic of the ASR data instead of keywords alone:

* G / OG / AS (the assist line written right after a goal) -> Goal; YC / RC / Y2C -> Card;
* the penalty kick (PSG, PM, "penalty save", or a highlight saying the penalty was saved/missed) -> Penalty, and so
  is every line that mentions the penalty within ``penalty_lead_min`` minutes before that kick (award, VAR check,
  "steps up"): commentators talk about a penalty from the moment it is given (README 2.2);
* a line that mentions a penalty with no kick around it ("penalty shout... the referee isn't interested",
  season statistics, "edge of the penalty area") stays No-Event: these are real hard negatives.

Text is made ASR-like (no "!!!", no stray symbols; ``noise_frac`` of the lines also get ASR noise).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from ..data.types import Sample
from .text_utils import asr_noise, asr_normalize

TYPE_TO_CLASS = {"G": "Goal", "OG": "Goal", "AS": "Goal", "YC": "Card", "RC": "Card", "Y2C": "Card",
                 "PSG": "Penalty", "PM": "Penalty", "penalty save": "Penalty"}
PEN_MENTION = re.compile(r"penalt|spot[- ]kick|the spot\b|twelve yards|12 yards", re.I)
PEN_KICK_TEXT = re.compile(r"penalty (is )?(saved|missed)|(saves?|miss(es|ed)?|denied) (from )?the (spot|penalty)|"
                           r"(saves?|misses) the penalty", re.I)
NOT_A_PENALTY = re.compile(r"penalty (area|box)|edge of the (penalty|box)|penalty spot\b(?! kick)", re.I)


def clean(text: str) -> str:
    text = text.replace("�", "'").replace("’", "'")
    text = re.sub(r"([!?.])\1+", r"\1", text)
    text = re.sub(r"\bgo+a+l+(s?)\b", r"goal\1", text, flags=re.I)  # "goooaaalll" -> "goal" (Whisper writes "goal")
    return asr_normalize(text)


def _dedupe(rows: list[dict]) -> list[dict]:
    """Drop exact duplicates and truncated copies (a line that is a prefix of another line of the same minute)."""
    out = []
    by_min: dict[float, list[str]] = {}
    for r in sorted(rows, key=lambda r: -len(r["Comment"])):
        c = r["Comment"].strip()
        if any(o.startswith(c) for o in by_min.get(r["Minute"], [])):
            continue
        by_min.setdefault(r["Minute"], []).append(c)
        out.append(r)
    return sorted(out, key=lambda r: (r["Minute"], r["_order"]))


def label_match(rows: list[dict], penalty_lead_min: float = 2.0) -> list[str]:
    kicks = [r["Minute"] for r in rows
             if r["Type"] in ("PSG", "PM", "penalty save") or (r["Type"] in ("highlight", "var", "comment")
                                                                 and PEN_KICK_TEXT.search(r["Comment"]))]
    labels = []
    for r in rows:
        lab = TYPE_TO_CLASS.get(r["Type"], "No-Event")
        if r["Type"] in ("highlight", "var", "comment"):
            mentions = PEN_MENTION.search(r["Comment"]) and not NOT_A_PENALTY.search(r["Comment"])
            if mentions and any(k - penalty_lead_min <= r["Minute"] <= k for k in kicks):
                lab = "Penalty"
        labels.append(lab)
    return labels


def load_livetext_samples(root: str | Path, seed: int = 42, noise_frac: float = 0.5,
                          penalty_lead_min: float = 2.0) -> list[Sample]:
    import pandas as pd

    root = Path(root)
    df = pd.concat([pd.read_parquet(p) for p in sorted(root.glob("*.parquet"))], ignore_index=True)
    df = df.dropna(subset=["Comment", "Minute"])
    df["_order"] = range(len(df))
    rng = random.Random(seed)
    out: list[Sample] = []
    for match, g in df.groupby("URL Code", sort=True):
        rows = _dedupe(g.to_dict("records"))
        labels = label_match(rows, penalty_lead_min)
        texts = [clean(r["Comment"]) for r in rows]
        for i, (r, lab) in enumerate(zip(rows, labels)):
            if len(texts[i].split()) < 3:
                continue
            noisy = rng.random() < noise_frac
            t = asr_noise(texts[i], rng) if noisy else texts[i]
            out.append(Sample(text=t, label=lab, source="livetext", ctx_before=texts[i - 1] if i else "",
                              ctx_after=texts[i + 1] if i + 1 < len(texts) else "", game=f"livetext:{match}",
                              half=1 if r["Minute"] <= 45 else 2, seg_idx=i, start=float(r["Minute"]) * 60,
                              end=float(r["Minute"]) * 60, split="train",
                              meta={"type": r["Type"], "minute": r["Minute"], "noisy": noisy,
                                    "hard_negative": bool(lab == "No-Event" and PEN_MENTION.search(r["Comment"]))}))
    return out


def summarize(samples: list[Sample]) -> dict:
    return {"by_label": dict(Counter(s.label for s in samples)),
            "penalty_by_type": dict(Counter(s.meta["type"] for s in samples if s.label == "Penalty")),
            "penalty_mention_negatives": sum(s.meta["hard_negative"] for s in samples),
            "matches": len({s.game for s in samples})}
