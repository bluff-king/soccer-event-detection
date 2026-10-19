"""Ground-truth events from SoccerNet Labels-v2.

Two interchangeable sources:

* ``official``: ``<labels_root>/<game>/Labels-v2.json`` downloaded with the SoccerNet pip package
  (``SoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=[...])``). Needs access to
  the SoccerNet servers (works on Colab).
* ``mirror``: the per-game ``event.json`` files committed to the ``sushant`` branch of
  https://github.com/SoccerNet/sn-echoes (``summaries/<game>/event.json``). They contain the
  Labels-v2 event list rendered as text ("13:10 Event: Goal by Chelsea") for 434 of the 500 labelled
  games, with 1-second resolution. Works everywhere GitHub is reachable.

``auto`` uses the official file when present and falls back to the mirror.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..labels import map_soccernet_label
from .types import Event

_MIRROR_LINE = re.compile(r"^(\d+):(\d\d) Event: (.+?)(?: by (.+))?$")
_HALF_HDR = {"First Half:": 1, "Second Half:": 2}


def parse_official(path: str | Path) -> list[Event]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for a in data.get("annotations", []):
        half = int(a["gameTime"].split(" - ")[0])
        t = float(a["position"]) / 1000.0
        out.append(Event(half, t, a["label"], map_soccernet_label(a["label"]), a.get("team", "")))
    return out


def parse_mirror_text(text: str) -> list[Event]:
    out = []
    half = 0
    for raw in text.split("\n"):
        line = raw.strip()
        if line in _HALF_HDR:
            half = _HALF_HDR[line]
            continue
        m = _MIRROR_LINE.match(line)
        if not m or half == 0:
            continue
        t = int(m.group(1)) * 60 + int(m.group(2))
        label = m.group(3).strip()
        out.append(Event(half, float(t), label, map_soccernet_label(label), (m.group(4) or "").strip()))
    return out


def parse_mirror(path: str | Path) -> list[Event]:
    with open(path, encoding="utf-8") as f:
        return parse_mirror_text(json.load(f)["input"])


def dedupe_events(events: list[Event], window: float = 5.0) -> tuple[list[Event], int]:
    """Drop repeated annotations of the same class within `window` seconds (a known Labels-v2
    artefact, e.g. "Goal" annotated twice at the same second). Returns (events, n_dropped)."""
    events = sorted(events, key=lambda e: (e.half, e.time))
    kept: list[Event] = []
    dropped = 0
    for e in events:
        dup = any(
            k.half == e.half and k.label == e.label and abs(k.time - e.time) <= window
            for k in kept[-10:]
        )
        if dup:
            dropped += 1
        else:
            kept.append(e)
    return kept, dropped


class LabelStore:
    def __init__(self, mirror_root: str | Path | None = None, official_root: str | Path | None = None,
                 source: str = "auto", dedupe_window: float = 5.0):
        self.mirror_root = Path(mirror_root) if mirror_root else None
        self.official_root = Path(official_root) if official_root else None
        self.source = source
        self.dedupe_window = dedupe_window
        self.n_duplicates = 0

    def _official_path(self, game: str) -> Path | None:
        if self.official_root is None:
            return None
        p = self.official_root / game / "Labels-v2.json"
        return p if p.exists() else None

    def _mirror_path(self, game: str) -> Path | None:
        if self.mirror_root is None:
            return None
        p = self.mirror_root / "summaries" / game / "event.json"
        return p if p.exists() else None

    def has(self, game: str) -> bool:
        return self.path(game) is not None

    def path(self, game: str) -> tuple[str, Path] | None:
        if self.source in ("auto", "official"):
            p = self._official_path(game)
            if p:
                return "official", p
            if self.source == "official":
                return None
        p = self._mirror_path(game)
        return ("mirror", p) if p else None

    def load(self, game: str) -> list[Event] | None:
        r = self.path(game)
        if r is None:
            return None
        kind, p = r
        events = parse_official(p) if kind == "official" else parse_mirror(p)
        if self.dedupe_window > 0:
            events, d = dedupe_events(events, self.dedupe_window)
            self.n_duplicates += d
        return events
