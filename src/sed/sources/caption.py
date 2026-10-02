"""SoccerNet-Caption as an extra (train-only) source.

* ``official``: ``<root>/<game>/Labels-caption.json`` (SoccerNet downloader, task caption-2023).
  Each annotation has ``label`` (e.g. "goal", "y-card", "penalty", "comments") and ``description``.
* ``mirror``: ``caption_event.json`` from the ``sushant`` branch of sn-echoes — Labels-v2 events and
  captions interleaved as text ("11:01 Caption: ..."). Captions there have no class, so a caption gets
  class ``c`` when it is the first caption (preferably one mentioning the class keyword) published
  within ``[t - 10 s, t + 90 s]`` of a Labels-v2 event of class ``c``; every other caption is
  No-Event (many of them mention "goal"/"penalty area" without an event -> natural hard negatives).

SoccerNet-Caption covers the *same games* as SoccerNet-v2, so captions are kept only for games in the
train split; valid/test games are dropped to avoid leakage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..data.align import mentions
from ..data.soccernet_labels import parse_mirror_text
from ..data.types import Sample
from ..labels import DEFAULT_PRIORITY
from .text_utils import asr_normalize, split_sentences, strip_scores

_CAP_LINE = re.compile(r"^(\d+):(\d\d) Caption: (.+)$")
_HALF_HDR = {"First Half:": 1, "Second Half:": 2}
OFFICIAL_MAP = {"goal": "Goal", "y-card": "Card", "r-card": "Card", "yr-card": "Card", "penalty": "Penalty"}


def parse_mirror_captions(text: str) -> list[tuple[int, float, str]]:
    out, half = [], 0
    for raw in text.split("\n"):
        line = raw.strip()
        if line in _HALF_HDR:
            half = _HALF_HDR[line]
            continue
        m = _CAP_LINE.match(line)
        if m and half:
            out.append((half, float(int(m.group(1)) * 60 + int(m.group(2))), m.group(3).strip()))
    return out


def label_mirror_captions(text: str, before: float = 10.0, after: float = 90.0) -> list[tuple[int, float, str, str]]:
    caps = parse_mirror_captions(text)
    events = [e for e in parse_mirror_text(text) if e.cls]
    labels = ["No-Event"] * len(caps)
    rank = {c: i for i, c in enumerate(DEFAULT_PRIORITY)}
    for e in sorted(events, key=lambda e: rank[e.cls]):
        cand = [i for i, (h, t, _) in enumerate(caps) if h == e.half and e.time - before <= t <= e.time + after
                and labels[i] == "No-Event"]
        if not cand:
            continue
        kw = [i for i in cand if mentions(e.cls, caps[i][2])]
        labels[(kw or cand)[0]] = e.cls
    return [(h, t, txt, lab) for (h, t, txt), lab in zip(caps, labels)]


def _to_sample(game: str, half: int, t: float, desc: str, label: str, style: str) -> Sample:
    desc = asr_normalize(strip_scores(desc))
    desc = re.sub(r"\bGOAL\b", "Goal", desc)
    if style == "sentence_window":
        sents = split_sentences(desc) or [desc]
        text, after = sents[0], " ".join(sents[1:3])
    else:
        text, after = desc, ""
    return Sample(text=text, label=label, source="caption", ctx_after=after, game=game, half=half,
                  start=t, end=t, split="train")


def load_caption_samples(games: list[str], mirror_root: str | Path | None = None,
                         official_root: str | Path | None = None, style: str = "sentence_window") -> list[Sample]:
    out: list[Sample] = []
    for g in games:
        off = Path(official_root) / g / "Labels-caption.json" if official_root else None
        if off is not None and off.exists():
            data = json.loads(off.read_text(encoding="utf-8"))
            for a in data.get("annotations", []):
                half = int(a["gameTime"].split(" - ")[0])
                lab = OFFICIAL_MAP.get(a.get("label", ""), "No-Event")
                out.append(_to_sample(g, half, float(a["position"]) / 1000, a.get("description", ""), lab, style))
            continue
        mir = Path(mirror_root) / "summaries" / g / "caption_event.json" if mirror_root else None
        if mir is not None and mir.exists():
            text = json.loads(mir.read_text(encoding="utf-8"))["input"]
            for half, t, desc, lab in label_mirror_captions(text):
                out.append(_to_sample(g, half, t, desc, lab, style))
    return out
