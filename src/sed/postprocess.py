"""From per-segment class probabilities to events.

1. temporal smoothing (moving average over neighbouring segments of the same half);
2. per-class thresholds -> candidate segments (class with the largest p_c / thr_c among those above);
3. temporal NMS: consecutive/nearby candidates of the same class (within ``nms_window`` s) are merged
   into one event (the most confident segment is kept) - several adjacent windows describing the same
   goal are a major source of segment-level "false positives";
4. event time = segment start - class offset (expected commentator delay).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .labels import CLASSES, EVENT_CLASSES, LABEL2ID, NO_EVENT
from .metrics import Detection, event_metrics, GTEvent


def smooth_probs(probs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(probs) == 0:
        return probs
    k = np.ones(window) / window
    pad = window // 2
    padded = np.pad(probs, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(probs.shape[1])], axis=1)
    return out


def threshold_predict(probs: np.ndarray, thresholds: dict[str, float] | None) -> np.ndarray:
    """Segment labels. Without thresholds: argmax. With thresholds: an event class wins if its
    probability reaches its threshold (largest p/thr ratio), otherwise No-Event."""
    if not thresholds:
        return probs.argmax(1)
    pred = np.full(len(probs), NO_EVENT, dtype=int)
    best_ratio = np.zeros(len(probs))
    for c in EVENT_CLASSES:
        i = LABEL2ID[c]
        thr = max(thresholds.get(c, 0.5), 1e-6)
        ratio = probs[:, i] / thr
        win = (probs[:, i] >= thr) & (ratio > best_ratio)
        pred[win] = i
        best_ratio[win] = ratio[win]
    return pred


def nms_1d(cands: list[Detection], window: float) -> list[Detection]:
    """Greedy temporal NMS per (game, half, class): keep the most confident detection and suppress
    others closer than ``window`` seconds."""
    groups: dict = defaultdict(list)
    for d in cands:
        groups[(d.game, d.half, d.cls)].append(d)
    out = []
    for g in groups.values():
        g.sort(key=lambda d: -d.confidence)
        kept: list[Detection] = []
        for d in g:
            if all(abs(d.time - k.time) >= window for k in kept):
                kept.append(d)
        out += kept
    return sorted(out, key=lambda d: (d.game, d.half, d.time))


def detect_events(meta: list[dict], probs: np.ndarray, thresholds: dict[str, float] | None,
                  smoothing: int = 1, nms_window: float = 30.0, offsets: dict[str, float] | None = None) -> list[Detection]:
    """``meta``: one dict per row with game, half, start, seg_idx (rows of a half must be contiguous
    and time-ordered, as produced by the window builder)."""
    offsets = offsets or {}
    groups: dict = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["game"], m["half"])].append(i)
    cands = []
    for (game, half), idx in groups.items():
        idx = sorted(idx, key=lambda i: meta[i]["start"])
        p = smooth_probs(probs[idx], smoothing)
        pred = threshold_predict(p, thresholds)
        for k, (i, y) in enumerate(zip(idx, pred)):
            if y == NO_EVENT:
                continue
            c = CLASSES[y]
            cands.append(Detection(game, half, c, float(meta[i]["start"]) - offsets.get(c, 0.0), float(p[k, y]),
                                   int(meta[i].get("seg_idx", -1))))
    return nms_1d(cands, nms_window)


def tune_thresholds(meta: list[dict], probs: np.ndarray, gts: list[GTEvent], tolerance: float, smoothing: int,
                    nms_window: float, offsets: dict[str, float] | None = None,
                    grid: np.ndarray | None = None) -> dict[str, float]:
    """Per-class threshold maximising event-level F1 on validation (classes tuned independently;
    other classes disabled with threshold > 1 while tuning one)."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = {}
    for c in EVENT_CLASSES:
        scores = []
        for t in grid:
            thr = {k: (t if k == c else 1.01) for k in EVENT_CLASSES}
            dets = detect_events(meta, probs, thr, smoothing, nms_window, offsets)
            m = event_metrics(dets, [g for g in gts if g.cls == c], tolerance)
            scores.append((m[c]["f1"], m[c]["precision"], t))
        best[c] = float(max(scores)[2])
    return best


def pr_curve_points(meta: list[dict], probs: np.ndarray, gts: list[GTEvent], tolerance: float, smoothing: int,
                    nms_window: float, offsets: dict[str, float] | None = None,
                    grid: np.ndarray | None = None) -> dict[str, list[tuple[float, float, float]]]:
    """Event-level (threshold, precision, recall) per class."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    out = {}
    for c in EVENT_CLASSES:
        pts = []
        for t in grid:
            thr = {k: (t if k == c else 1.01) for k in EVENT_CLASSES}
            m = event_metrics(detect_events(meta, probs, thr, smoothing, nms_window, offsets),
                              [g for g in gts if g.cls == c], tolerance)
            pts.append((float(t), m[c]["precision"], m[c]["recall"]))
        out[c] = pts
    return out
