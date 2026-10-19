"""Post-processing sweep on saved probabilities (CPU only, no training): temporal smoothing x NMS window.
Thresholds are re-tuned on validation for every setting; the setting is chosen on validation event F1 and its
test score is reported (test is never used for a choice). Thresholds are tuned per class independently, so the
sweep also picks a (smoothing, NMS) pair per class on validation ("per_class") and evaluates that combination.

    python scripts/sweep_postprocess.py outputs/local/v2_delays_views_hardneg --out docs/results/postprocess_sweep.json
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from sed.data.prepare import load_split
from sed.evaluate import evaluate_probs

SMOOTHING = [1, 3, 5]
NMS = [20.0, 30.0, 45.0, 60.0]


def sweep(run: Path) -> dict:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    samples = {s: load_split(cfg, s) for s in ("valid", "test")}
    probs = {s: np.load(run / f"probs_{s}.npy") for s in samples}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for sm in SMOOTHING:
            for nms in NMS:
                c = {**cfg, "postprocess": {**cfg["postprocess"], "smoothing": sm, "nms_window": nms,
                                            "thresholds": None}}
                r = evaluate_probs(c, probs, samples, Path(tmp))
                row = {"smoothing": sm, "nms_window": nms, "thresholds": r["thresholds"]}
                for s in ("valid", "test"):
                    e = r[s]["event"]
                    row[s] = {"P": e["overall"]["precision"], "R": e["overall"]["recall"], "F1": e["overall"]["f1"],
                              **{c_: e[c_]["f1"] for c_ in ("Goal", "Card", "Penalty")}}
                rows.append(row)
                print(f"  {run.name} smooth={sm} nms={nms:4.0f}  valid F1 {row['valid']['F1']:.3f}  "
                      f"test F1 {row['test']['F1']:.3f}", flush=True)
    best = max(rows, key=lambda x: x["valid"]["F1"])
    default = next(x for x in rows if x["smoothing"] == cfg["postprocess"]["smoothing"]
                   and x["nms_window"] == cfg["postprocess"]["nms_window"])
    # per-class choice on validation, then one evaluation of the combination
    pick = {c: max(rows, key=lambda x: x["valid"][c]) for c in ("Goal", "Card", "Penalty")}
    sm = {c: r["smoothing"] for c, r in pick.items()}
    nms = {c: r["nms_window"] for c, r in pick.items()}
    with tempfile.TemporaryDirectory() as tmp:
        c = {**cfg, "postprocess": {**cfg["postprocess"], "smoothing": sm, "nms_window": nms, "thresholds": None}}
        r = evaluate_probs(c, probs, samples, Path(tmp))
    pc = {"smoothing": sm, "nms_window": nms, "thresholds": r["thresholds"]}
    for s_ in ("valid", "test"):
        e = r[s_]["event"]
        pc[s_] = {"P": e["overall"]["precision"], "R": e["overall"]["recall"], "F1": e["overall"]["f1"],
                  **{c_: e[c_]["f1"] for c_ in ("Goal", "Card", "Penalty")}}
    return {"rows": rows, "chosen_on_valid": best, "default": default, "per_class": pc}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="docs/results/postprocess_sweep.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for r in a.runs:
        res[Path(r).name] = sweep(Path(r))
        b, d = res[Path(r).name]["chosen_on_valid"], res[Path(r).name]["default"]
        print(f"{Path(r).name}: default (s={d['smoothing']}, nms={d['nms_window']}) valid {d['valid']['F1']:.3f} "
              f"test {d['test']['F1']:.3f} -> chosen (s={b['smoothing']}, nms={b['nms_window']}) "
              f"valid {b['valid']['F1']:.3f} test {b['test']['F1']:.3f}", flush=True)
        pc = res[Path(r).name]["per_class"]
        print(f"   per-class smoothing {pc['smoothing']} nms {pc['nms_window']}: valid {pc['valid']['F1']:.3f} "
              f"test {pc['test']['F1']:.3f} (test Goal/Card/Pen {pc['test']['Goal']:.3f} / {pc['test']['Card']:.3f} / "
              f"{pc['test']['Penalty']:.3f})", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
