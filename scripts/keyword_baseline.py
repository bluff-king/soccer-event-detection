"""Non-neural reference: a window is an event if its *current* segment contains a class keyword
(priority Penalty > Card > Goal). Runs on CPU in seconds on the real valid/test sets and shows how
much of the problem is keyword spotting (and how many false positives keywords alone produce).

    python scripts/keyword_baseline.py --config configs/base.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sed.config import load_config
from sed.data.align import mentions
from sed.data.prepare import load_split
from sed.evaluate import evaluate_probs, summary_row
from sed.labels import LABEL2ID


def keyword_probs(samples):
    p = np.zeros((len(samples), 4), dtype=np.float32)
    for i, s in enumerate(samples):
        for c in ("Penalty", "Card", "Goal"):
            if mentions(c, s.text):
                p[i, LABEL2ID[c]] = 0.9
                p[i, 0] = 0.1
                break
        else:
            p[i, 0] = 1.0
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="outputs/keyword_baseline")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    cfg["postprocess"]["thresholds"] = {"Goal": 0.5, "Card": 0.5, "Penalty": 0.5}  # binary scores: nothing to tune
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = {s: load_split(cfg, s) for s in ("valid", "test")}
    probs = {s: keyword_probs(v) for s, v in samples.items()}
    res = evaluate_probs(cfg, probs, samples, out)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    row = summary_row("keyword_rule", res, "valid")
    row.update({f"test_{k}": v for k, v in summary_row("", res, "test").items() if k != "run"})
    row["train_time_min"] = 0
    (out / "summary.json").write_text(json.dumps(row, indent=1))
    print(json.dumps(row, indent=1))
    print("FP groups (valid):", json.dumps(res["valid"]["fp_diagnosis"]["groups"], indent=1))


if __name__ == "__main__":
    main()
