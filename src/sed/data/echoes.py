"""Loader for SoccerNet-Echoes ASR transcripts (https://github.com/SoccerNet/sn-echoes).

Layout (main branch):  Dataset/<variant>/<league>/<season>/<game>/{1,2}_asr.json
with variant in whisper_v1, whisper_v1_en, whisper_v2, whisper_v2_en, whisper_v3, whisper_v3_en.
`*_en` folders only exist for non-English commentary (Google-translated to English).

The `sushant` branch ships `SN-echoes-lang.csv` (language + recommended whisper version per half),
which we use for the `auto_en` / `auto` variants.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from .types import Segment

VARIANTS = ["whisper_v1", "whisper_v1_en", "whisper_v2", "whisper_v2_en", "whisper_v3", "whisper_v3_en"]


@lru_cache(maxsize=4)
def load_lang_table(path: str) -> dict[str, tuple[str, str]]:
    """Map 'game/half' -> (lang, selected version e.g. 'v2'). Empty dict if the file is missing."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["Game/Half"]] = (row["Lang"].strip(), row["selected"].strip())
    return out


def read_asr_json(path: str | Path) -> list[Segment]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data["segments"]
    out = []
    # keys are segment indices, but a few translated files have translated keys ("eleven"), so
    # order by start time instead of by key
    for v in segs.values():
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        start, end, text = v[0], v[1], v[2]
        text = (text or "").strip() if isinstance(text, str) else ""
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if text:
            out.append(Segment(start, end, text))
    out.sort(key=lambda s: (s.start, s.end))
    return out


def resolve_variant(game: str, half: int, variant: str, lang_table: dict) -> list[str]:
    """Return candidate variant folders (in order of preference) for one half."""
    if variant in VARIANTS:
        return [variant]
    lang, sel = lang_table.get(f"{game}/{half}", ("", "v2"))
    base = f"whisper_{sel}"
    others = [f"whisper_v{v}" for v in (2, 3, 1) if f"whisper_v{v}" != base]
    if variant == "auto_en":
        if lang == "en":
            return [base] + others
        # translated version first; fall back to the original if the translation is missing
        return [base + "_en"] + [o + "_en" for o in others] + ([base] if not lang else [])
    if variant == "auto":
        return [base] + others
    raise ValueError(f"unknown ASR variant {variant!r}")


class EchoesCorpus:
    """Index of available transcripts in a local clone of sn-echoes."""

    def __init__(self, root: str | Path, variant: str = "auto_en", lang_csv: str | Path | None = None):
        self.root = Path(root)
        self.dataset = self.root / "Dataset"
        self.variant = variant
        self.lang_table = load_lang_table(str(lang_csv)) if lang_csv else {}

    def list_games(self) -> list[str]:
        games = set()
        for v in VARIANTS:
            d = self.dataset / v
            if not d.exists():
                continue
            for p in d.glob("*/*/*"):
                if p.is_dir():
                    games.add(str(p.relative_to(d)))
        return sorted(games)

    def lang(self, game: str, half: int) -> str:
        return self.lang_table.get(f"{game}/{half}", ("", ""))[0]

    def load(self, game: str, half: int) -> list[Segment] | None:
        for v in resolve_variant(game, half, self.variant, self.lang_table):
            p = self.dataset / v / game / f"{half}_asr.json"
            if p.exists():
                return read_asr_json(p)
        return None
