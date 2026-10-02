import numpy as np
import pytest

from sed.metrics import Detection, GTEvent, balanced_subset_indices, event_metrics, segment_metrics
from sed.model.losses import FocalLoss, class_weights_from_counts
from sed.postprocess import detect_events, nms_1d, smooth_probs, threshold_predict, tune_thresholds


def test_segment_metrics_perfect_and_highlight():
    y = [0, 1, 2, 3, 0]
    m = segment_metrics(y, y)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["highlight_f1"] == 1
    # one Goal predicted as Card, one No-Event predicted as Goal
    m = segment_metrics([0, 1, 2, 0], [0, 2, 2, 1])
    assert m["confusion_matrix"][1][2] == 1
    # highlight: tp=1 (Card), predicted events=3, gold events=2 -> P=1/3, R=1/2
    assert m["highlight_precision"] == pytest.approx(1 / 3)
    assert m["highlight_recall"] == pytest.approx(1 / 2)
    assert m["per_class"]["Card"]["precision"] == pytest.approx(0.5)


def test_event_matching_tolerance_one_to_one():
    gts = [GTEvent("g", 1, "Goal", 100.0), GTEvent("g", 1, "Card", 300.0)]
    dets = [Detection("g", 1, "Goal", 110.0, 0.9), Detection("g", 1, "Goal", 120.0, 0.8),  # duplicate -> FP
            Detection("g", 1, "Card", 400.0, 0.7),  # too far -> FP, Card missed
            Detection("g", 2, "Goal", 100.0, 0.9)]  # other half -> FP
    m = event_metrics(dets, gts, tolerance=30)
    assert m["Goal"]["tp"] == 1 and m["Goal"]["fp"] == 2
    assert m["Card"]["fn"] == 1 and m["Card"]["fp"] == 1
    assert m["overall"]["recall"] == pytest.approx(0.5)
    assert event_metrics(dets, gts, tolerance=150)["Card"]["tp"] == 1


def test_nms_merges_adjacent_detections():
    d = [Detection("g", 1, "Goal", t, c) for t, c in [(10, 0.5), (12, 0.9), (15, 0.6), (80, 0.7)]]
    kept = nms_1d(d, 30)
    assert [(k.time, k.confidence) for k in kept] == [(12, 0.9), (80, 0.7)]


def test_threshold_predict_and_smoothing():
    p = np.array([[0.7, 0.3, 0.0, 0.0], [0.6, 0.1, 0.3, 0.0], [0.9, 0.0, 0.0, 0.1]])
    assert threshold_predict(p, None).tolist() == [0, 0, 0]
    assert threshold_predict(p, {"Goal": 0.25, "Card": 0.25, "Penalty": 0.5}).tolist() == [1, 2, 0]
    s = smooth_probs(np.eye(4)[[1, 0, 0, 0, 0]].astype(float), 3)
    assert s.shape == (5, 4) and s[1, 1] == pytest.approx(1 / 3)


def test_detect_events_offsets_and_tuning():
    meta = [{"game": "g", "half": 1, "start": float(t), "seg_idx": i} for i, t in enumerate(range(0, 200, 2))]
    probs = np.tile([0.97, 0.01, 0.01, 0.01], (100, 1))
    probs[50:54] = [0.2, 0.8, 0.0, 0.0]  # Goal talk at 100-106 s
    dets = detect_events(meta, probs, {"Goal": 0.5, "Card": 0.5, "Penalty": 0.5}, 1, 30, {"Goal": 5.0})
    assert len(dets) == 1 and dets[0].cls == "Goal" and 95 <= dets[0].time <= 101
    gts = [GTEvent("g", 1, "Goal", 98.0)]
    thr = tune_thresholds(meta, probs, gts, 30, 1, 30, {"Goal": 5.0})
    assert 0.05 < thr["Goal"] <= 0.8


def test_balanced_subset_and_losses():
    y = np.array([0] * 100 + [1] * 5 + [3] * 5)
    idx = balanced_subset_indices(y, 2.0)
    assert (y[idx] != 0).sum() == 10 and (y[idx] == 0).sum() == 20
    w = class_weights_from_counts(["No-Event"] * 90 + ["Goal"] * 10)
    assert w[1] > w[0]
    import torch

    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    fl = FocalLoss(2.0)(logits, torch.tensor([0, 1]))
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    assert 0 < fl.item() < ce.item()
