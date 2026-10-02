"""Full evaluation of a trained model: segment level + event level on valid/test, threshold tuning
on valid, PR curves, NMS ablation and false-positive diagnosis. Writes ``metrics.json`` and figures
into the run directory and the inference settings into the checkpoint (``sed_inference.json``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.prepare import class_offsets, load_events, load_split
from .diagnose import diagnose_false_positives
from .labels import CLASSES, EVENT_CLASSES, LABEL2ID
from .metrics import balanced_subset_indices, event_metrics, segment_metrics
from .model.predict import predict_probs
from .postprocess import detect_events, pr_curve_points, threshold_predict, tune_thresholds


def _meta(samples):
    return [{"game": s.game, "half": s.half, "start": s.start, "seg_idx": s.seg_idx} for s in samples]


def _plot_pr(curves: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for c, pts in curves.items():
        r = [p[2] for p in pts]
        p = [p[1] for p in pts]
        ax.plot(r, p, marker="o", ms=3, label=c)
    ax.set_xlabel("recall (event level)")
    ax.set_ylabel("precision (event level)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cm(cm, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_probs(cfg: dict, probs: dict[str, np.ndarray], samples: dict[str, list], out_dir: Path,
                   thresholds: dict | None = None) -> dict:
    pp, ev = cfg["postprocess"], cfg["eval"]
    tol, smooth, nms = ev["tolerance"], pp["smoothing"], pp["nms_window"]
    offsets = class_offsets(cfg)
    gts = {s: load_events(cfg, s) for s in samples}
    metas = {s: _meta(samples[s]) for s in samples}
    if thresholds is None:
        thresholds = pp.get("thresholds")
    if not thresholds and pp.get("tune_thresholds", True) and "valid" in samples:
        thresholds = tune_thresholds(metas["valid"], probs["valid"], gts["valid"], tol, smooth, nms, offsets)
    thresholds = thresholds or {c: 0.5 for c in EVENT_CLASSES}
    res: dict = {"thresholds": thresholds, "offsets": offsets, "smoothing": smooth, "nms_window": nms,
                 "tolerance": tol}
    for s in samples:
        y = np.array([LABEL2ID[x.label] for x in samples[s]])
        p = probs[s]
        y_arg = p.argmax(1)
        y_thr = threshold_predict(p, thresholds)
        r = {"segment_argmax": segment_metrics(y, y_arg), "segment_thresholded": segment_metrics(y, y_thr)}
        idx = balanced_subset_indices(y, ev.get("balanced_neg_ratio", 1.0), cfg.get("seed", 42))
        r["segment_balanced_argmax"] = segment_metrics(y[idx], y_arg[idx])
        dets = detect_events(metas[s], p, thresholds, smooth, nms, offsets)
        r["event"] = event_metrics(dets, gts[s], tol)
        # ablations of post-processing (same thresholds)
        r["event_no_nms"] = event_metrics(detect_events(metas[s], p, thresholds, smooth, 0.0, offsets), gts[s], tol)
        r["event_argmax_no_nms"] = event_metrics(detect_events(metas[s], p, None, 1, 0.0, offsets), gts[s], tol)
        texts = {(x.game, x.half, x.seg_idx): " | ".join([x.ctx_before, x.text, x.ctx_after]) for x in samples[s]}
        r["fp_diagnosis"] = diagnose_false_positives(dets, gts[s], tol, texts)
        r["n_gt_events"] = len(gts[s])
        res[s] = r
        _plot_cm(r["segment_argmax"]["confusion_matrix"], out_dir / f"confusion_{s}.png", f"{s} (argmax)")
    if "valid" in samples:
        curves = pr_curve_points(metas["valid"], probs["valid"], gts["valid"], tol, smooth, nms, offsets)
        res["pr_curves_valid"] = curves
        _plot_pr(curves, out_dir / "pr_curves_valid.png", "valid: event-level PR vs threshold")
    return res


def evaluate_checkpoint(cfg: dict, model, tok, out_dir: str | Path, splits=("valid", "test"),
                        ckpt_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = cfg["train"]
    samples = {s: load_split(cfg, s) for s in splits}
    probs = {}
    for s in splits:
        probs[s] = predict_probs(model, tok, samples[s], cfg["model"]["max_length"], tr["eval_batch_size"],
                                 fp16=tr.get("fp16", True))
        np.save(out_dir / f"probs_{s}.npy", probs[s])
    res = evaluate_probs(cfg, probs, samples, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    if ckpt_dir:
        write_inference_config(cfg, res, ckpt_dir)
    return res


def write_inference_config(cfg: dict, res: dict, ckpt_dir: str | Path) -> None:
    d = cfg["data"]
    info = {"thresholds": res["thresholds"], "offsets": res["offsets"], "smoothing": res["smoothing"],
            "nms_window": res["nms_window"], "ctx_before": d["ctx_before"], "ctx_after": d["ctx_after"],
            "max_length": cfg["model"]["max_length"], "classes": CLASSES}
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(ckpt_dir) / "sed_inference.json").write_text(json.dumps(info, indent=1))


def summary_row(name: str, res: dict, split: str = "valid") -> dict:
    r = res[split]
    return {
        "run": name,
        "seg_acc": r["segment_argmax"]["accuracy"],
        "seg_macro_f1": r["segment_argmax"]["macro_f1"],
        "seg_highlight_f1": r["segment_argmax"]["highlight_f1"],
        "bal_acc": r["segment_balanced_argmax"]["accuracy"],
        "bal_highlight_f1": r["segment_balanced_argmax"]["highlight_f1"],
        "evt_precision": r["event"]["overall"]["precision"],
        "evt_recall": r["event"]["overall"]["recall"],
        "evt_f1": r["event"]["overall"]["f1"],
        **{f"evt_f1_{c}": r["event"][c]["f1"] for c in EVENT_CLASSES},
    }
