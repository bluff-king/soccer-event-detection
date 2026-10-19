"""Re-run the evaluation of finished runs from their saved probabilities (no GPU, no training), e.g. after a
change of the post-processing code such as the threshold grid. Keeps the training fields of metrics.json /
summary.json and rewrites the evaluation fields, the figures and best/sed_inference.json.

    python scripts/reevaluate.py outputs/local/baseline outputs/local/v2_baseline [--set postprocess.smoothing=3]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from sed.config import _merge, load_config
from sed.data.prepare import load_split
from sed.evaluate import evaluate_probs, summary_row, write_inference_config

TRAIN_FIELDS = ("train_time_s", "n_train", "train_class_counts", "train_source_counts", "reused_predictions_from")


def reevaluate(run_dir: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if overrides:  # apply dotted overrides on top of the saved config
        tmp = run_dir / "_reeval_override.yaml"
        tmp.write_text(yaml.safe_dump(cfg))
        cfg = load_config(tmp, overrides)
        tmp.unlink()
    samples = {s: load_split(cfg, s) for s in ("valid", "test")}
    probs = {s: np.load(run_dir / f"probs_{s}.npy") for s in samples}
    res = evaluate_probs(cfg, probs, samples, run_dir)
    old = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    res.update({k: old[k] for k in TRAIN_FIELDS if k in old})
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    if (run_dir / "best").exists():
        write_inference_config(cfg, res, run_dir / "best")
    old_row = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    row = summary_row(cfg.get("run_name", run_dir.name), res, "valid")
    row.update({f"test_{k}": v for k, v in summary_row("", res, "test").items() if k != "run"})
    row["train_time_min"] = old_row.get("train_time_min", round(res.get("train_time_s", 0) / 60, 1))
    (run_dir / "summary.json").write_text(json.dumps(row, indent=1))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    for r in a.runs:
        row = reevaluate(Path(r), a.set)
        print(r, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()
                             if k.startswith(("evt", "test_evt_f1"))}), flush=True)


if __name__ == "__main__":
    main()
