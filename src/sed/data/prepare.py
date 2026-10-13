"""Prepare clean data files under ``data.processed_dir``:

    echoes_{train,valid,test}.jsonl   Echoes windows (natural distribution)
    caption.jsonl                     SoccerNet-Caption samples (train games only)
    asr_views.jsonl                   other Whisper transcripts of the train games (real ASR, same labels)
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
from ..sources.asr_views import load_asr_views
from ..sources.caption import load_caption_samples
from ..sources.kaggle_events import load_kaggle_samples, overlapping_kaggle_games
from ..sources.synthetic import SynthConfig, generate_hard_negatives, generate_synthetic, mine_echoes_hard_negatives
from .build import build_echoes, class_counts, load_jsonl, save_jsonl, subsample_negatives
from .types import Sample

SPLITS = ("train", "valid", "test")
# settings baked into the labels/windows of a processed dir: a config that differs in any of them needs its own
# data.processed_dir (otherwise it would silently train on labels made with other settings)
LABELING_KEYS = ("delay_min", "delay_max", "delay_per_class", "ctx_before", "ctx_after", "asr_variant", "label_source",
                 "split_method", "max_games", "dedupe_window")


def labeling_signature(cfg: dict) -> dict:
    return {k: cfg["data"].get(k) for k in LABELING_KEYS}


def check_labeling(cfg: dict) -> None:
    """Raise if the processed dir was built with other labelling settings than ``cfg``."""
    path = processed(cfg, "manifest.json")
    if not path.exists():
        return
    built = json.loads(path.read_text()).get("labeling")
    if built is None:  # built before the signature existed
        return
    want = labeling_signature(cfg)
    diff = {k: (built.get(k), want[k]) for k in want if built.get(k) != want[k]}
    if diff:
        raise ValueError(f"{cfg['data']['processed_dir']} was prepared with other labelling settings "
                         f"(built, wanted): {diff}. Set a separate data.processed_dir and run scripts/prepare_data.py")


def processed(cfg: dict, name: str) -> Path:
    return Path(cfg["data"]["processed_dir"]) / name


def prepare(cfg: dict, sources: list[str] | None = None) -> dict:
    """Build every file. ``sources`` limits which external sources are (re)built
    (default: all of asr_views, caption, kaggle, synthetic, hardneg)."""
    d = cfg["data"]
    sources = sources if sources is not None else ["asr_views", "caption", "kaggle", "synthetic", "hardneg"]
    seed = cfg.get("seed", 42)
    b = build_echoes(cfg)
    split_games = {s: sorted({g.game for g in b.games if g.split == s}) for s in SPLITS}
    heldout = set(split_games["valid"]) | set(split_games["test"])
    manifest: dict = {"labeling": labeling_signature(cfg),
                      "games": {s: len(v) for s, v in split_games.items()}, "split_games": split_games,
                      "label_duplicates_removed": b.n_label_duplicates, "counts": {}}
    for s in SPLITS:
        rows = [x for x in b.samples if x.split == s]
        save_jsonl(rows, processed(cfg, f"echoes_{s}.jsonl"))
        manifest["counts"][f"echoes_{s}"] = class_counts(rows)

    for s in SPLITS:  # ground-truth highlight events (for event-level evaluation)
        evs = [{"game": g.game, "half": e.half, "cls": e.cls, "time": e.time, "label": e.label}
               for g in b.games if g.split == s for e in g.events if e.cls and e.half in g.halves]
        p = processed(cfg, f"events_{s}.jsonl")
        p.write_text("".join(json.dumps(x) + "\n" for x in evs), encoding="utf-8")

    train_echoes = [x for x in b.samples if x.split == "train"]
    if "asr_views" in sources:
        av = load_asr_views(cfg, [g for g in b.games if g.split == "train"],
                            cfg["sources"].get("asr_views", {}).get("mode", "en"))
        assert not ({x.game for x in av} & heldout), "ASR-view leakage into valid/test"
        save_jsonl(av, processed(cfg, "asr_views.jsonl"))
        manifest["counts"]["asr_views"] = class_counts(av)
        manifest["asr_views_variants"] = dict(Counter(x.meta["variant"] for x in av))

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
    check_labeling(cfg)
    return load_jsonl(processed(cfg, f"echoes_{split}.jsonl"))


def load_events(cfg: dict, split: str):
    from ..metrics import GTEvent

    out = []
    with processed(cfg, f"events_{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(GTEvent(d["game"], d["half"], d["cls"], d["time"]))
    return out


def class_offsets(cfg: dict) -> dict[str, float]:
    """Expected commentator delay per class = centre of its labelling window (used to map a detected
    segment back to the event time)."""
    from .build import per_class_delays

    d = cfg["data"]
    pc = per_class_delays(cfg) or {}
    return {c: (pc.get(c, (d["delay_min"], d["delay_max"]))[0] + pc.get(c, (d["delay_min"], d["delay_max"]))[1]) / 2
            for c in CLASSES[1:]}


def assemble_train(cfg: dict) -> list[Sample]:
    """Echoes train (optionally negative-subsampled) + enabled external sources. Valid/test are
    never touched here: external data only ever enters the training set."""
    d = cfg["data"]
    seed = cfg.get("seed", 42)
    train = load_split(cfg, "train")
    train = subsample_negatives(train, d.get("train_neg_ratio"), seed, d.get("keep_keyword_negatives", True))
    valid_test_games = {s.game for sp in ("valid", "test") for s in load_split(cfg, sp)}
    for name in ("asr_views", "caption", "kaggle", "synthetic", "hardneg"):
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
        if sc.get("neg_ratio"):  # natural-distribution sources (ASR views): same negative subsampling as Echoes
            extra = subsample_negatives(extra, sc["neg_ratio"], seed, d.get("keep_keyword_negatives", True))
        if sc.get("max_samples"):
            random.Random(seed).shuffle(extra)
            extra = extra[: sc["max_samples"]]
        train += extra
    return train
