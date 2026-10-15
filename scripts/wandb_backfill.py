"""Upload finished runs (outputs/<run>/summary.json + metrics.json + history.json + figures) to Weights &
Biases, e.g. runs trained before tracking was enabled. Runs already logged (``.wandb_synced``) are skipped.

    WANDB_API_KEY=... python scripts/wandb_backfill.py outputs/* [--set logging.entity=... logging.group=...]
"""

import argparse
import json
from pathlib import Path

import yaml

from sed.config import _merge
from sed.tracking import Tracker


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--set", nargs="*", default=[], help="logging overrides, e.g. logging.entity=team")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    over: dict = {"logging": {"wandb": True}}
    for kv in a.set:
        k, _, v = kv.partition("=")
        node = over
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    for d in map(Path, a.run_dirs):
        if not ((d / "summary.json").exists() and (d / "metrics.json").exists() and (d / "config.yaml").exists()):
            continue
        if (d / ".wandb_synced").exists() and not a.force:
            print(f"[skip] {d} already on wandb")
            continue
        cfg = _merge(yaml.safe_load((d / "config.yaml").read_text()), over)
        t = Tracker(cfg, job_type="backfill")
        if t.run is None:
            print("[wandb] not available, stopping")
            return
        hist = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
        for rec in hist:
            t.log({f"valid_select/{k}": v for k, v in rec.items() if k not in ("step", "tag")}, step=rec["step"])
        res = json.loads((d / "metrics.json").read_text())
        t.log_results(res, d, json.loads((d / "summary.json").read_text()))
        (d / ".wandb_synced").write_text(t.run.url or "")
        print(f"[ok] {d} -> {t.run.url}")
        t.finish()


if __name__ == "__main__":
    main()
