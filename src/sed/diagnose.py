"""False-positive diagnosis at event level.

Every unmatched detection is assigned to the first matching group:

* ``duplicate``     - a same-class GT event within +-T is already matched by another detection
                      (the same event reported twice; NMS window too small);
* ``time_offset``   - a same-class GT event exists within (T, 3T]: right event, wrong window;
* ``wrong_class``   - a GT event of another highlight class within +-T;
* ``replay_recall`` - a same-class GT event happened earlier in the half (> 3T before) or the text
                      contains replay / back-reference cues ("again", "replay", "earlier", ...);
* ``keyword_no_event`` - the window mentions the class keyword but no event is near: either a true
                      hard negative (near miss, "should be a card") or label noise (missing GT);
* ``other``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .data.align import mentions
from .metrics import Detection, GTEvent, match_events

_REPLAY = re.compile(r"\b(replay|again|earlier|first half|before|last (week|season|game|time)|already|"
                     r"once more|look at|another look|watch)\b", re.IGNORECASE)


def diagnose_false_positives(dets: list[Detection], gts: list[GTEvent], tolerance: float,
                             texts: dict[tuple, str] | None = None, max_examples: int = 8) -> dict:
    """``texts`` maps (game, half, seg_idx) -> window text (for keyword/replay cues and examples)."""
    det_ok, gt_ok = match_events(dets, gts, tolerance)
    by_half = defaultdict(list)
    for j, g in enumerate(gts):
        by_half[(g.game, g.half)].append(j)
    texts = texts or {}
    counts: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    for d, ok in zip(dets, det_ok):
        if ok:
            continue
        txt = texts.get((d.game, d.half, d.seg_idx), "")
        cands = [gts[j] for j in by_half[(d.game, d.half)]]
        same = [g for g in cands if g.cls == d.cls]
        if any(abs(g.time - d.time) <= tolerance for g in same):
            cat = "duplicate"
        elif any(tolerance < abs(g.time - d.time) <= 3 * tolerance for g in same):
            cat = "time_offset"
        elif any(g.cls != d.cls and abs(g.time - d.time) <= tolerance for g in cands):
            cat = "wrong_class"
        elif any(d.time - g.time > 3 * tolerance for g in same) or _REPLAY.search(txt):
            cat = "replay_recall"
        elif txt and mentions(d.cls, txt):
            cat = "keyword_no_event"
        else:
            cat = "other"
        counts[cat] += 1
        if len(examples[cat]) < max_examples:
            examples[cat].append({"game": d.game, "half": d.half, "time": round(d.time, 1), "cls": d.cls,
                                  "confidence": round(d.confidence, 3), "text": txt[:300]})
    n_fp = sum(counts.values())
    return {"n_detections": len(dets), "n_false_positives": n_fp,
            "groups": {k: {"count": v, "share": v / n_fp if n_fp else 0.0} for k, v in counts.most_common()},
            "examples": dict(examples),
            "missed_events": sum(1 for ok in gt_ok if not ok)}


def segment_fp_vs_event(meta: list[dict], y_true, y_pred, dets: list[Detection]) -> dict:
    """How many segment-level FPs disappear once adjacent windows are merged into events."""
    seg_fp = sum(1 for t, p in zip(y_true, y_pred) if p != 0 and p != t)
    return {"segment_level_fp": int(seg_fp), "event_level_detections": len(dets)}
