"""Keep the model checkpoint of the best run(s) only (each xlm-roberta-base checkpoint is ~1 GB).

Runs are ranked by validation event F1 (summary.json -> evt_f1). For every other finished run the weights in
best/ are deleted; best/sed_inference.json, probs_*.npy, metrics.json and the figures stay, so the run can still be
re-evaluated (scripts/reevaluate.py) or used as a two-stage verifier (sed.two_stage only reads probs_*.npy).
Runs without summary.json (still training) are never touched.

    python scripts/prune_checkpoints.py "outputs/local/v2_*" [--keep 1] [--dry-run]
"""

import argparse
import glob
import json
import shutil
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="run directories or glob patterns")
    ap.add_argument("--keep", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    runs = sorted({Path(p) for pat in a.patterns for p in glob.glob(pat) if Path(p).is_dir()})
    done = [(json.loads((r / "summary.json").read_text()).get("evt_f1", -1.0), r) for r in runs
            if (r / "summary.json").exists()]
    done.sort(key=lambda x: -x[0])
    keep = {r for _, r in done[: a.keep]}
    freed = 0
    for f1, r in done:
        best = r / "best"
        tag = "keep" if r in keep else "prune"
        size = sum(p.stat().st_size for p in best.rglob("*") if p.is_file()) if best.exists() else 0
        print(f"{tag:5s} evt_f1={f1:.3f} {size / 2**30:5.2f} GB  {r}")
        if r in keep or not best.exists():
            continue
        freed += size
        if not a.dry_run:
            info = best / "sed_inference.json"
            saved = info.read_text() if info.exists() else None
            shutil.rmtree(best, ignore_errors=True)
            if saved is not None:
                best.mkdir(exist_ok=True)
                (best / "sed_inference.json").write_text(saved)
    print(f"{'would free' if a.dry_run else 'freed'} {freed / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
