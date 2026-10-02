"""Prepare clean data files under ``data.processed_dir``:

    echoes_{train,valid,test}.jsonl   Echoes windows (natural distribution)
    caption.jsonl                     SoccerNet-Caption samples (train games only)
    kaggle.jsonl                      Kaggle Football Events (masked, overlapping valid/test games removed)
    synthetic.jsonl / hardneg.jsonl   generated samples (source + meta per sample)
    manifest.json                     games per split, counts per source/class, leakage checks

and assemble the training set for a given config (``assemble_train``).
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from ..labels import CLASSES
from ..sources.caption import load_caption_samples
from ..sources.kaggle_events import load_kaggle_samples, overlapping_kaggle_games
from ..sources.synthetic import SynthConfig, generate_hard_negatives, generate_synthetic, mine_echoes_hard_negatives
from .build import build_echoes, class_counts, load_jsonl, save_jsonl, subsample_negatives
from .types import Sample

SPLITS = ("train", "valid", "test")


def processed(cfg: dict, name: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / name


def prepare(cfg: dict, sources: list[str] | None = None) -> dict:
    """Build every file. ``sources`` limits which external sources are (re)built
    (default: all of caption, kaggle, synthetic, hardneg)."""
    d = cfg["data"]
    sources = sources if sources is not None else ["caption", "kaggle", "synthetic", "hardneg"]
    seed = cfg.get("seed", 42)
    b = build_echoes(cfg)
    split_games = {s: sorted({g.game for g in b.games if g.split == s}) for s in SPLITS}
    heldout = set(split_games["valid"]) | set(split_games["test"])
    manifest: dict = {"games": {s: len(v) for s, v in split_games.items()}, "split_games": split_games,
                      "label_duplicates_removed": b.n_label_duplicates, "counts": {}}
    for s in SPLITS:
        rows = [x for x in b.samples if x.split == s]
        save_jsonl(rows, processed(cfg, f"echoes_{s}.jsonl"))
        manifest["counts"][f"echoes_{s}"] = class_counts(rows)

    train_echoes = [x for x in b.samples if x.split == "train"]
    if "caption" in sources:
        caps = load_caption_samples(split_games["train"], d.get("mirror_root"), d.get("official_root"),
                                    cfg["sources"]["caption"].get("style", "sentence_window"))
        assert not ({c.game for c in caps} & heldout), "caption leakage into valid/test"
        save_jsonl(caps, processed(cfg, "caption.jsonl"))
        manifest["counts"]["caption"] = class_counts(caps)

    if "kaggle" in sources:
        kroot = Path(d["raw_root"]) / "football-events"
        if (kroot / "events.csv").exists():
            kc = cfg["sources"]["kaggle"]
            with (kroot / "ginf.csv").open(newline="", encoding="utf-8", errors="replace") as f:
                ginf = list(csv.DictReader(f))
            excl = overlapping_kaggle_games(ginf, sorted(heldout))
            ks = load_kaggle_samples(kroot, excl, kc.get("max_per_class"), kc.get("neg_per_pos", 1.0), seed,
                                     kc.get("mask", True))
            save_jsonl(ks, processed(cfg, "kaggle.jsonl"))
            manifest["counts"]["kaggle"] = class_counts(ks)
            manifest["kaggle_games_removed_overlap_valid_test"] = len(excl)

    sc = cfg["sources"].get("synthetic", {})
    synth_cfg = SynthConfig(backend=sc.get("backend", "template"), targets=sc.get("targets"),
                            hardneg_per_category=cfg["sources"].get("hardneg", {}).get("per_category", 300),
                            per_call=sc.get("per_call", 5), noise=sc.get("noise", True), seed=seed,
                            backend_kwargs=sc.get("backend_kwargs"))
    if "synthetic" in sources:
        seeds: dict[str, list[str]] = {c: [] for c in CLASSES[1:]}
        for x in train_echoes:  # real event windows from TRAIN games only
            if x.label in seeds:
                seeds[x.label].append(" ".join([x.ctx_before, x.text, x.ctx_after]))
        cap_path = processed(cfg, "caption.jsonl")
        if cap_path.exists():
            for x in load_jsonl(cap_path):
                if x.label in seeds:
                    seeds[x.label].append(x.text + " " + x.ctx_after)
        syn = generate_synthetic(seeds, synth_cfg)
        save_jsonl(syn, processed(cfg, "synthetic.jsonl"))
        manifest["counts"]["synthetic"] = class_counts(syn)

    if "hardneg" in sources:
        hc = cfg["sources"].get("hardneg", {})
        hn = generate_hard_negatives(synth_cfg)
        ev_times = {(g.game, e.half): [] for g in b.games for e in g.events}
        for g in b.games:
            for e in g.events:
                if e.cls:
                    ev_times[(g.game, e.half)].append((e.time, e.cls))
        hn += mine_echoes_hard_negatives(train_echoes, hc.get("mined_max", 5000), seed, ev_times,
                                         hc.get("mined_min_gap", 120.0))
        save_jsonl(hn, processed(cfg, "hardneg.jsonl"))
        manifest["counts"]["hardneg"] = class_counts(hn)
        manifest["hardneg_categories"] = dict(Counter(x.meta.get("category", "?") for x in hn))

    path = processed(cfg, "manifest.json")
    old = json.loads(path.read_text()) if path.exists() else {}
    old_counts = old.get("counts", {})
    old_counts.update(manifest["counts"])
    manifest["counts"] = old_counts
    path.write_text(json.dumps({**old, **manifest}, indent=1))
    return manifest


def load_split(cfg: dict, split: str) -> list[Sample]:
    return load_jsonl(processed(cfg, f"echoes_{split}.jsonl"))


def assemble_train(cfg: dict) -> list[Sample]:
    """Echoes train (optionally negative-subsampled) + enabled external sources. Valid/test are
    never touched here: external data only ever enters the training set."""
    d = cfg["data"]
    seed = cfg.get("seed", 42)
    train = load_split(cfg, "train")
    train = subsample_negatives(train, d.get("train_neg_ratio"), seed, d.get("keep_keyword_negatives", True))
    valid_test_games = {s.game for sp in ("valid", "test") for s in load_split(cfg, sp)}
    for name in ("caption", "kaggle", "synthetic", "hardneg"):
        sc = cfg["sources"].get(name, {})
        if not sc.get("enabled"):
            continue
        p = processed(cfg, f"{name}.jsonl")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run scripts/prepare_data.py first")
        extra = load_jsonl(p)
        leaked = {s.game for s in extra} & valid_test_games
        if leaked:
            raise AssertionError(f"{name}: {len(leaked)} valid/test games in external data")
        if sc.get("max_samples"):
            random.Random(seed).shuffle(extra)
            extra = extra[: sc["max_samples"]]
        train += extra
    return train
