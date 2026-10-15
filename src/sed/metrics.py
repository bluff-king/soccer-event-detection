"""Evaluation at two levels.

Segment level: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix and
"highlight F1" = micro-F1 over the event classes (Goal/Card/Penalty) with No-Event as the negative
class (a prediction counts as correct only with the right class).

Event level: detections ``{type, time, confidence}`` (after post-processing) are matched one-to-one
to ground-truth events of the same class within +-T seconds (greedy by confidence). Reports
precision / recall / F1 per class and overall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import CLASSES, EVENT_CLASSES, NO_EVENT


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def confusion(y_true, y_pred, n: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def segment_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion(y_true, y_pred)
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r, f = _prf(tp, fp, fn)
        per[c] = {"precision": p, "recall": r, "f1": f, "support": int(cm[i, :].sum())}
    ev = [i for i, c in enumerate(CLASSES) if c in EVENT_CLASSES]
    tp = sum(cm[i, i] for i in ev)
    pred_ev = sum(cm[:, i].sum() for i in ev)
    gold_ev = sum(cm[i, :].sum() for i in ev)
    hp, hr, hf = _prf(tp, pred_ev - tp, gold_ev - tp)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(np.mean([per[c]["f1"] for c in CLASSES])),
        "highlight_precision": hp, "highlight_recall": hr, "highlight_f1": hf,
        "highlight_macro_f1": float(np.mean([per[c]["f1"] for c in EVENT_CLASSES])),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
    }


@dataclass
class Detection:
    game: str
    half: int
    cls: str
    time: float
    confidence: float
    seg_idx: int = -1


@dataclass
class GTEvent:
    game: str
    half: int
    cls: str
    time: float


def match_events(dets: list[Detection], gts: list[GTEvent], tolerance: float) -> tuple[list[bool], list[bool]]:
    """Greedy one-to-one matching (highest confidence first). Returns (det_matched, gt_matched)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].confidence)
    gt_by_key: dict = {}
    for j, g in enumerate(gts):
        gt_by_key.setdefault((g.game, g.half, g.cls), []).append(j)
    gt_used = [False] * len(gts)
    det_ok = [False] * len(dets)
    for i in order:
        d = dets[i]
        best, best_dt = None, None
        for j in gt_by_key.get((d.game, d.half, d.cls), []):
            if gt_used[j]:
                continue
            dt = abs(gts[j].time - d.time)
            if dt <= tolerance and (best_dt is None or dt < best_dt):
                best, best_dt = j, dt
        if best is not None:
            gt_used[best] = True
            det_ok[i] = True
    return det_ok, gt_used


def event_metrics(dets: list[Detection], gts: list[GTEvent], tolerance: float = 30.0) -> dict:
    det_ok, gt_ok = match_events(dets, gts, tolerance)
    out = {"tolerance": tolerance}
    tp_all = fp_all = fn_all = 0
    for c in EVENT_CLASSES:
        tp = sum(1 for d, ok in zip(dets, det_ok) if d.cls == c and ok)
        fp = sum(1 for d, ok in zip(dets, det_ok) if d.cls == c and not ok)
        fn = sum(1 for g, ok in zip(gts, gt_ok) if g.cls == c and not ok)
        p, r, f = _prf(tp, fp, fn)
        out[c] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    p, r, f = _prf(tp_all, fp_all, fn_all)
    out["overall"] = {"precision": p, "recall": r, "f1": f, "tp": tp_all, "fp": fp_all, "fn": fn_all}
    out["macro_f1"] = float(np.mean([out[c]["f1"] for c in EVENT_CLASSES]))
    return out


def balanced_subset_indices(y_true, neg_ratio: float = 1.0, seed: int = 0) -> np.ndarray:
    """Indices of all positives + ``neg_ratio`` x as many random negatives (for metrics comparable to
    a reference evaluated on a class-balanced set)."""
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pos = np.where(y != NO_EVENT)[0]
    neg = np.where(y == NO_EVENT)[0]
    k = min(len(neg), int(len(pos) * neg_ratio))
    return np.sort(np.concatenate([pos, rng.choice(neg, size=k, replace=False)]))
