"""Mean +- std over seeds of runs named <config>_s<seed> (e.g. outputs/local/v3/*), validation and test.

    python scripts/summarize_seeds.py "outputs/local/v3/*" --out docs/results/v3_seeds.md
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = [("evt_f1", "valid F1"), ("test_evt_precision", "test P"), ("test_evt_recall", "test R"), ("test_evt_f1", "test F1"),
        ("test_evt_f1_Goal", "test Goal"), ("test_evt_f1_Card", "test Card"), ("test_evt_f1_Penalty", "test Pen")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    groups = defaultdict(list)
    for pat in a.patterns:
        for d in sorted(glob.glob(pat)):
            m = re.match(r"(.+)_s(\d+)$", Path(d).name)
            if m and (Path(d) / "summary.json").exists():
                groups[m.group(1)].append(json.loads((Path(d) / "summary.json").read_text()))
    lines = ["| config | seeds | " + " | ".join(k[1] for k in KEYS) + " | min/run |", "|" + "---|" * (len(KEYS) + 3)]
    for g, rows in groups.items():
        cells = []
        for k, _ in KEYS:
            v = np.array([r[k] for r in rows])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}")
        t = np.mean([r.get("train_time_min", 0) for r in rows])
        lines.append(f"| {g} | {len(rows)} | " + " | ".join(cells) + f" | {t:.0f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("Mean ± sample std over seeds (validation F1 is optimistic: used for selection).\n\n"
                               + text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
