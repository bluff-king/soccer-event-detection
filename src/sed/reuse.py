"""Post-processing-only experiments: re-evaluate the saved probabilities of an already trained run
with different post-processing (smoothing, NMS, thresholds, tolerance) instead of training the same
model again. Used when a config sets ``reuse_probs_from: <run dir>``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .data.prepare import load_split
from .evaluate import evaluate_probs, summary_row, write_inference_config

# training-relevant settings that must match the source run (everything that changes the model)
_TRAIN_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "warmup_ratio", "grad_accum", "loss", "class_weights",
               "focal_gamma", "sampler", "max_steps", "select_metric", "select_eval_negatives")


def _enabled_sources(cfg: dict) -> list[str]:
    return sorted(k for k, v in (cfg.get("sources") or {}).items() if isinstance(v, dict) and v.get("enabled"))


def incompatibility(cfg: dict, src_cfg: dict) -> str | None:
    """Why the source run's predictions cannot stand in for training ``cfg`` (None = compatible)."""
    if cfg.get("seed", 42) != src_cfg.get("seed", 42):
        return "seed differs"
    if cfg["data"] != src_cfg["data"]:
        diff = sorted(k for k in set(cfg["data"]) | set(src_cfg["data"]) if cfg["data"].get(k) != src_cfg["data"].get(k))
        return f"data settings differ: {diff}"
    if _enabled_sources(cfg) != _enabled_sources(src_cfg):
        return "external sources differ"
    if cfg["model"] != src_cfg["model"]:
        return "model differs"
    diff = [k for k in _TRAIN_KEYS if cfg["train"].get(k) != src_cfg["train"].get(k)]
    if diff:
        return f"training settings differ: {diff}"
    return None


def try_reuse(cfg: dict) -> dict | None:
    """Evaluate ``cfg`` from the source run's saved probabilities. Returns the results, or None (with a
    printed reason) when the source run is missing or not equivalent, in which case the caller trains."""
    src = cfg.get("reuse_probs_from")
    if not src:
        return None
    src = Path(src)
    need = [src / "config.yaml", src / "probs_valid.npy", src / "probs_test.npy"]
    if not all(p.exists() for p in need):
        print(f"[reuse] {src} has no saved predictions yet -> training instead", flush=True)
        return None
    src_cfg = yaml.safe_load((src / "config.yaml").read_text())
    why = incompatibility(cfg, src_cfg)
    if why:
        print(f"[reuse] cannot reuse {src}: {why} -> training instead", flush=True)
        return None
    samples = {s: load_split(cfg, s) for s in ("valid", "test")}
    probs = {s: np.load(src / f"probs_{s}.npy") for s in samples}
    for s in samples:
        if len(probs[s]) != len(samples[s]):
            print(f"[reuse] {src}: {s} predictions do not match the data -> training instead", flush=True)
            return None

    from .tracking import Tracker

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[reuse] post-processing only: evaluating predictions of {src} (no training)", flush=True)
    tracker = Tracker(cfg, job_type="postprocess")
    res = evaluate_probs(cfg, probs, samples, out)
    res["train_time_s"] = 0.0
    res["reused_predictions_from"] = str(src)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    # the model is the source run's checkpoint; store these post-processing settings next to the results
    write_inference_config(cfg, res, out)
    row = summary_row(cfg.get("run_name", out.name), res, "valid")
    row.update({f"test_{k}": v for k, v in summary_row("", res, "test").items() if k != "run"})
    row["train_time_min"] = 0.0
    (out / "summary.json").write_text(json.dumps(row, indent=1))
    print("[summary]", json.dumps(row), flush=True)
    tracker.summary({"reused_predictions_from": str(src)})
    tracker.log_results(res, out, row)
    if tracker.run is not None:
        (out / ".wandb_synced").write_text(tracker.run.url or "")
    tracker.finish()
    return res
