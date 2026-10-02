"""Data survey: count real samples per class for every source and write docs/DATA_SURVEY.md (+ figures).

    python scripts/survey_data.py --config configs/base.yaml [--out docs]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from sed.config import load_config
from sed.data.align import mention_profile, mentions_any
from sed.data.build import build_echoes, class_counts
from sed.labels import CLASSES
from sed.sources.caption import load_caption_samples
from sed.sources.kaggle_events import kaggle_class_counts, overlapping_kaggle_games

SPLITS = ("train", "valid", "test")


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="docs")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    # ---------------- Echoes + Labels-v2 ----------------
    b = build_echoes(cfg)
    games_by_split = {s: [g for g in b.games if g.split == s] for s in SPLITS}
    raw_events = {s: Counter() for s in SPLITS}
    for s, gs in games_by_split.items():
        for g in gs:
            for e in g.events:
                if e.cls:
                    raw_events[s][e.label] += 1
    seg_counts = {s: class_counts([x for x in b.samples if x.split == s]) for s in SPLITS}
    lang = Counter()
    from sed.data.echoes import EchoesCorpus

    corpus = EchoesCorpus(d["echoes_root"], d["asr_variant"], d["lang_csv"])
    for g in b.games:
        for h in g.halves:
            lang[corpus.lang(g.game, h) or "?"] += 1
    stats["echoes"] = {
        "games": {s: len(v) for s, v in games_by_split.items()},
        "skipped": dict(b.skipped),
        "label_duplicates_removed": b.n_label_duplicates,
        "raw_events": {s: dict(v) for s, v in raw_events.items()},
        "segment_windows": seg_counts,
        "languages_halves": dict(lang.most_common()),
    }

    # Label noise: number of Goal annotations vs the score in the game name
    mism = []
    for g in b.games:
        m = re.search(r" (\d+) - (\d+) ", g.game.split("/")[-1])
        n_goal = sum(1 for e in g.events if e.label == "Goal")
        if m and n_goal != int(m.group(1)) + int(m.group(2)):
            mism.append((g.game, n_goal, int(m.group(1)) + int(m.group(2))))
    stats["label_noise_goal_vs_score"] = {"games": len(mism), "examples": mism[:8]}

    # Hard negatives already present in Echoes: No-Event windows mentioning a class keyword
    hn = Counter()
    for x in b.samples:
        if x.label == "No-Event":
            for c in mentions_any(x.text):
                hn[(x.split, c)] += 1
    stats["echoes_keyword_negatives"] = {f"{s}/{c}": n for (s, c), n in sorted(hn.items())}

    # Commentator reaction profile (keyword mentions around events)
    lo, hi, bs = -150, 150, 5
    profiles = {}
    for c in ("Goal", "Card", "Penalty"):
        tot = np.zeros(int((hi - lo) / bs))
        for g in b.games:
            for h, segs in g.halves.items():
                tot += np.array(mention_profile(segs, g.events, h, c, lo, hi, bs))
        profiles[c] = tot.tolist()
    stats["mention_profile"] = {"lo": lo, "hi": hi, "bin": bs, "counts": profiles}
    first = {}
    for c in ("Goal", "Card", "Penalty"):
        dd = [o.delay for o in b.delays if o.cls == c]
        hit = np.array([x for x in dd if x is not None])
        first[c] = {"events": len(dd), "with_mention": int(len(hit)),
                    "pct": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                    np.percentile(hit, [10, 25, 50, 75, 90]).round(1).tolist())) if len(hit) else {}}
    stats["first_mention_delay"] = first

    # ---------------- SoccerNet-Caption ----------------
    all_games = [g.game for g in b.games]
    train_games = [g.game for g in games_by_split["train"]]
    cap_all = load_caption_samples(all_games, d.get("mirror_root"), d.get("official_root"))
    cap_train = [x for x in cap_all if x.game in set(train_games)]
    stats["caption"] = {"all_games": class_counts(cap_all), "train_games_only": class_counts(cap_train),
                        "n_games_with_captions": len({x.game for x in cap_all}),
                        "keyword_negatives_train": sum(1 for x in cap_train if x.label == "No-Event" and mentions_any(x.text))}

    # ---------------- Kaggle Football Events ----------------
    kroot = Path(d["raw_root"]) / "football-events"
    if (kroot / "events.csv").exists():
        with (kroot / "ginf.csv").open(newline="", encoding="utf-8", errors="replace") as f:
            ginf = list(csv.DictReader(f))
        heldout = [g.game for g in b.games if g.split in ("valid", "test")]
        ov_heldout = overlapping_kaggle_games(ginf, heldout)
        ov_all = overlapping_kaggle_games(ginf, all_games)
        stats["kaggle"] = {"games": len(ginf), "rows_per_class": kaggle_class_counts(kroot),
                           "overlap_with_soccernet_games": len(ov_all),
                           "overlap_with_valid_test_games(removed)": len(ov_heldout)}
    else:
        stats["kaggle"] = None

    (out / "data_survey.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    make_figures(stats, out / "figures")
    write_md(stats, cfg, out / "DATA_SURVEY.md")
    print((out / "DATA_SURVEY.md").read_text()[:6000])


def make_figures(stats: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Class distribution of Echoes windows per split (log scale)
    seg = stats["echoes"]["segment_windows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(CLASSES))
    for i, s in enumerate(SPLITS):
        ax.bar(x + (i - 1) * 0.27, [seg[s][c] for c in CLASSES], width=0.27, label=s)
    ax.set_yscale("log")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("# windows (log)")
    ax.set_title("SoccerNet-Echoes windows per class (delay-aware labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    # Mention profile: when do commentators mention the event keyword?
    mp = stats["mention_profile"]
    centers = np.arange(mp["lo"], mp["hi"], mp["bin"]) + mp["bin"] / 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
    for ax, c in zip(axes, ("Goal", "Card", "Penalty")):
        cnt = np.array(mp["counts"][c])
        base = np.median(np.r_[cnt[:4], cnt[-4:]])
        ax.bar(centers, cnt, width=mp["bin"] * 0.9, color="#4C78A8")
        ax.axhline(base, color="#E45756", ls="--", lw=1, label="baseline rate")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{c}: keyword mentions vs. label time")
        ax.set_xlabel("segment start - event time (s)")
    axes[0].set_ylabel("# mentions")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "delay_profile.png", dpi=120)
    plt.close(fig)


def write_md(stats: dict, cfg: dict, path: Path) -> None:
    e = stats["echoes"]
    d = cfg["data"]
    L = ["# Data survey (auto-generated by `scripts/survey_data.py`)", ""]
    L += [f"Config: ASR variant `{d['asr_variant']}`, labels `{d['label_source']}`, split `{d['split_method']}`, "
          f"delay window [{d['delay_min']}, {d['delay_max']}] s, context {d['ctx_before']}+1+{d['ctx_after']} segments.", ""]
    L += ["## 1. SoccerNet-Echoes x Labels-v2", ""]
    L += [f"Games with transcript + labels: " + ", ".join(f"{s} **{n}**" for s, n in e["games"].items())
          + f". Skipped: {e['skipped']}. Duplicate Labels-v2 annotations merged: {e['label_duplicates_removed']}.", ""]
    labs = ["Goal", "Yellow card", "Red card", "Yellow->red card", "Penalty"]
    L += ["### Ground-truth events (Labels-v2) in these games", ""]
    L += [md_table(["split"] + labs, [[s] + [e["raw_events"][s].get(l, 0) for l in labs] for s in SPLITS])]
    L += ["", "### Window samples per class (one per ASR segment)", ""]
    L += [md_table(["split"] + CLASSES + ["% events"],
                   [[s] + [e["segment_windows"][s][c] for c in CLASSES]
                    + [f"{100 * (1 - e['segment_windows'][s]['No-Event'] / max(1, sum(e['segment_windows'][s].values()))):.2f}%"]
                    for s in SPLITS])]
    L += ["", "![class distribution](figures/class_distribution.png)", ""]
    L += ["Halves per commentary language: " + ", ".join(f"{k}: {v}" for k, v in e["languages_halves"].items()), ""]
    L += ["## 2. Commentator delay", "", "![delay profile](figures/delay_profile.png)", ""]
    L += [md_table(["class", "#events", "with keyword mention in [-30,+90] s", "first-mention delay p10/p25/p50/p75/p90 (s)"],
                   [[c, v["events"], v["with_mention"], " / ".join(str(x) for x in v["pct"].values())]
                    for c, v in stats["first_mention_delay"].items()])]
    L += ["", "## 3. SoccerNet-Caption (mirror, labelled by alignment to Labels-v2)", ""]
    cap = stats["caption"]
    L += [md_table(["subset"] + CLASSES, [["all games"] + [cap["all_games"][c] for c in CLASSES],
                                         ["train games only (used)"] + [cap["train_games_only"][c] for c in CLASSES]])]
    L += ["", f"Train No-Event captions mentioning goal/card/penalty (natural hard negatives): {cap['keyword_negatives_train']}", ""]
    L += ["## 4. Kaggle Football Events", ""]
    k = stats["kaggle"]
    if k:
        L += [md_table(["games"] + CLASSES + ["overlap w/ SoccerNet", "overlap w/ valid+test (removed)"],
                       [[k["games"]] + [k["rows_per_class"][c] for c in CLASSES]
                        + [k["overlap_with_soccernet_games"], k["overlap_with_valid_test_games(removed)"]]])]
    else:
        L += ["not downloaded"]
    L += ["", "## 5. Natural hard negatives inside Echoes (No-Event windows mentioning a class keyword)", ""]
    L += [md_table(["split/keyword", "#windows"], [[k2, v] for k2, v in stats["echoes_keyword_negatives"].items()])]
    n = stats["label_noise_goal_vs_score"]
    L += ["", "## 6. Label noise", "",
          f"Games whose number of `Goal` annotations differs from the final score in the game name: **{n['games']}**.", ""]
    L += [md_table(["game", "#Goal labels", "goals in score"], [[g, a, s] for g, a, s in n["examples"]])]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
