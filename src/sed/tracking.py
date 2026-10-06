"""Optional Weights & Biases tracking. Disabled unless ``logging.wandb: true``; every call is a no-op
when wandb is missing, no API key is configured or wandb fails, so training never depends on it.

The API key is read by wandb from ``WANDB_API_KEY`` (the Colab notebook copies it from a Colab Secret).
"""

from __future__ import annotations

import os
from pathlib import Path

from .labels import EVENT_CLASSES


def run_display_name(cfg: dict) -> str:
    """e.g. ``colab_data_caption_xlmr-base_len128_ep5`` (same style as the existing project runs)."""
    m = cfg["model"]["name"].split("/")[-1].replace("xlm-roberta", "xlmr")
    prefix = cfg.get("logging", {}).get("name_prefix") or "run"
    return f"{prefix}_{cfg.get('run_name') or Path(cfg['output_dir']).name}_{m}_len{cfg['model']['max_length']}" \
           f"_ep{cfg['train']['epochs']}"


def flatten_results(res: dict, split: str) -> dict:
    """Scalar metrics of one split for a wandb summary."""
    r = res[split]
    out = {}
    for k in ("segment_argmax", "segment_thresholded", "segment_balanced_argmax"):
        for m in ("accuracy", "macro_f1", "highlight_precision", "highlight_recall", "highlight_f1"):
            out[f"{split}/{k}/{m}"] = r[k][m]
    for k in ("event", "event_no_nms", "event_argmax_no_nms"):
        for c in EVENT_CLASSES + ["overall"]:
            for m in ("precision", "recall", "f1"):
                out[f"{split}/{k}/{c}/{m}"] = r[k][c][m]
    for g, v in r["fp_diagnosis"]["groups"].items():
        out[f"{split}/fp/{g}"] = v["count"]
    return out


class Tracker:
    def __init__(self, cfg: dict, job_type: str = "train"):
        self.run = None
        lg = cfg.get("logging", {}) or {}
        if not lg.get("wandb"):
            return
        if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
            print("[wandb] logging.wandb is on but WANDB_API_KEY is not set -> not logging", flush=True)
            return
        try:
            import wandb

            self.run = wandb.init(project=lg.get("project") or "football-highlight", entity=lg.get("entity") or None,
                                  group=lg.get("group") or None, name=run_display_name(cfg), job_type=job_type,
                                  config=cfg, dir=cfg.get("output_dir"), reinit="finish_previous")
            print(f"[wandb] logging to {self.run.url}", flush=True)
        except Exception as e:  # never let tracking break training
            print(f"[wandb] disabled: {e}", flush=True)
            self.run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self.run is not None:
            try:
                self.run.log(data, step=step)
            except Exception as e:  # pragma: no cover
                print(f"[wandb] log failed: {e}", flush=True)

    def summary(self, data: dict) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(data)
            except Exception as e:  # pragma: no cover
                print(f"[wandb] summary failed: {e}", flush=True)

    def log_results(self, res: dict, out_dir: str | Path, row: dict | None = None) -> None:
        """Final metrics of a run: scalars per split, the summary row as a table, and the figures."""
        if self.run is None:
            return
        import wandb

        data = {}
        for s in ("valid", "test"):
            if s in res:
                data.update(flatten_results(res, s))
        data.update({f"thresholds/{c}": v for c, v in res.get("thresholds", {}).items()})
        self.summary(data)
        if row:
            self.summary({f"summary/{k}": v for k, v in row.items() if k != "run"})
            self.log({"results_table": wandb.Table(columns=list(row), data=[list(row.values())])})
        imgs = {p.stem: wandb.Image(str(p)) for p in sorted(Path(out_dir).glob("*.png"))}
        if imgs:
            self.log(imgs)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:  # pragma: no cover
                pass
            self.run = None
