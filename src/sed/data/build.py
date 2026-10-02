"""Build the Echoes window dataset (all splits) from raw transcripts + labels."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..labels import CLASSES
from .align import DelayObs, label_segments, measure_delays, mentions_any
from .echoes import EchoesCorpus
from .soccernet_labels import LabelStore
from .splits import assign_splits, check_disjoint
from .types import Event, Sample, Segment
from .windows import make_windows


@dataclass
class GameData:
    game: str
    split: str
    halves: dict[int, list[Segment]]
    events: list[Event]


@dataclass
class EchoesBuild:
    games: list[GameData]
    samples: list[Sample]
    delays: list[DelayObs] = field(default_factory=list)
    n_label_duplicates: int = 0
    skipped: Counter = field(default_factory=Counter)


def load_games(cfg: dict) -> tuple[list[GameData], Counter, int]:
    d = cfg["data"]
    corpus = EchoesCorpus(d["echoes_root"], d.get("asr_variant", "auto_en"), d.get("lang_csv"))
    store = LabelStore(d.get("mirror_root"), d.get("official_root"), d.get("label_source", "auto"),
                       d.get("dedupe_window", 5.0))
    games = corpus.list_games()
    split_of = assign_splits(games, d.get("split_method", "official"), seed=cfg.get("seed", 42))
    skipped: Counter = Counter()
    out: list[GameData] = []
    per_split: Counter = Counter()
    max_games = d.get("max_games")
    for g in games:
        split = split_of[g]
        if split == "unlabelled":
            skipped["no_split(challenge)"] += 1
            continue
        if max_games and per_split[split] >= max_games:
            continue
        events = store.load(g)
        if events is None:
            skipped["no_labels"] += 1
            continue
        halves = {}
        for h in (1, 2):
            segs = corpus.load(g, h)
            if segs:
                halves[h] = segs
        if not halves:
            skipped["no_transcript"] += 1
            continue
        out.append(GameData(g, split, halves, events))
        per_split[split] += 1
    check_disjoint({s: {x.game for x in out if x.split == s} for s in ("train", "valid", "test")})
    return out, skipped, store.n_duplicates


def build_echoes(cfg: dict, games: list[GameData] | None = None) -> EchoesBuild:
    d = cfg["data"]
    skipped: Counter = Counter()
    dups = 0
    if games is None:
        games, skipped, dups = load_games(cfg)
    samples: list[Sample] = []
    delays: list[DelayObs] = []
    for gd in games:
        for h, segs in gd.halves.items():
            labels = label_segments(segs, gd.events, h, d["delay_min"], d["delay_max"],
                                    per_class=per_class_delays(cfg))
            samples += make_windows(segs, labels, gd.game, h, d["ctx_before"], d["ctx_after"], gd.split)
            delays += measure_delays(segs, gd.events, h, gd.game)
    return EchoesBuild(games, samples, delays, dups, skipped)


def per_class_delays(cfg: dict) -> dict[str, tuple[float, float]] | None:
    pc = cfg["data"].get("delay_per_class")
    return {k: (float(v[0]), float(v[1])) for k, v in pc.items()} if pc else None


def subsample_negatives(samples: list[Sample], ratio: float | None, seed: int = 42,
                        keep_keyword_negatives: bool = True) -> list[Sample]:
    """Keep all positives and ``ratio`` x #positives No-Event windows (keyword-mentioning negatives,
    i.e. natural hard negatives, are always kept when ``keep_keyword_negatives``)."""
    if not ratio:
        return samples
    rng = random.Random(seed)
    pos = [s for s in samples if s.label != "No-Event"]
    neg = [s for s in samples if s.label == "No-Event"]
    hard = [s for s in neg if keep_keyword_negatives and mentions_any(s.text)]
    easy = [s for s in neg if not (keep_keyword_negatives and mentions_any(s.text))]
    n_easy = max(0, int(len(pos) * ratio) - len(hard))
    rng.shuffle(easy)
    out = pos + hard + easy[:n_easy]
    out.sort(key=lambda s: (s.game, s.half, s.seg_idx))
    return out


def class_counts(samples: list[Sample]) -> dict[str, int]:
    c = Counter(s.label for s in samples)
    return {k: c.get(k, 0) for k in CLASSES}


def save_jsonl(samples: list[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Sample]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Sample(**json.loads(line)))
    return out
