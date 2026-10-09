"""Run several configs sequentially and collect a comparison table.

    python scripts/run_experiments.py configs/base.yaml configs/ablation/data_*.yaml [--force] [--set train.epochs=1]

Writes outputs/results.md and outputs/results.csv (validation + test metrics per run). Runs whose
summary.json already exists are skipped unless --force (so a Colab session can be resumed).
"""

import argparse
import csv
import json
from pathlib import Path

from sed.config import load_config


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def collect(out_dirs, path_md="outputs/results.md", path_csv="outputs/results.csv"):
    rows = []
    for d in out_dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    if not rows:
        return rows
    cols = ["run", "seg_acc", "seg_macro_f1", "seg_highlight_f1", "bal_acc", "bal_highlight_f1", "evt_precision",
            "evt_recall", "evt_f1", "evt_f1_Goal", "evt_f1_Card", "evt_f1_Penalty", "test_evt_precision",
            "test_evt_recall", "test_evt_f1", "train_time_min"]
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    Path(path_md).write_text("Validation metrics unless prefixed with test_. seg_* = segment level (argmax, natural "
                             "distribution), bal_* = class-balanced subset, evt_* = event level (+-T s, tuned "
                             "thresholds + NMS).\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every config")
    a = ap.parse_args(argv)
    from sed.data.prepare import prepare, processed
    from sed.train import train

    out_dirs = []
    for c in a.configs:
        cfg = load_config(c, a.set)
        out_dirs.append(cfg["output_dir"])
        if (Path(cfg["output_dir"]) / "summary.json").exists() and not a.force:
            print(f"[skip] {c}: {cfg['output_dir']}/summary.json exists")
            continue
        if not processed(cfg, "echoes_train.jsonl").exists():
            prepare(cfg)
        print(f"===== {c} -> {cfg['output_dir']}", flush=True)
        train(cfg)
        collect(out_dirs)
    collect(out_dirs)


if __name__ == "__main__":
    main()
