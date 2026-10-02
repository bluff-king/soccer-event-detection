"""Two-stage detection to raise precision.

Stage 1 (recall): a trained run's segment probabilities with *low* per-class thresholds, chosen on
validation as the highest threshold that still reaches ``target_recall`` (event level), then NMS.

Stage 2 (verify): every candidate event is re-scored by a verifier and kept if the verifier agrees:

* ``model:<run_dir>`` - another trained run (e.g. a larger backbone or wider context). Its saved
  ``probs_<split>.npy`` are looked up for the candidate segment (windows are aligned by
  game/half/segment index); score = geometric mean of both probabilities; per-class threshold tuned
  on validation for event F1.
* ``llm:hf:<model>`` / ``llm:anthropic[:<model>]`` - an LLM reads +-k segments around the candidate
  and answers whether the event is happening *now* (not a replay, near miss or reference).

    python -m sed.two_stage --config outputs/baseline/config.yaml --stage1 outputs/baseline \
        --verifier model:outputs/xlmr_large --splits valid test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .config import load_config
from .data.prepare import class_offsets, load_events, load_split
from .labels import EVENT_CLASSES, LABEL2ID
from .metrics import Detection, event_metrics
from .postprocess import detect_events


def recall_thresholds(meta, probs, gts, tol, smooth, nms, offsets, target: float = 0.9) -> dict[str, float]:
    grid = np.round(np.arange(0.95, 0.0, -0.025), 3)
    out = {}
    for c in EVENT_CLASSES:
        chosen = float(grid[-1])
        for t in grid:  # from high to low: first threshold reaching the target recall
            thr = {k: (t if k == c else 1.01) for k in EVENT_CLASSES}
            m = event_metrics(detect_events(meta, probs, thr, smooth, nms, offsets), [g for g in gts if g.cls == c], tol)
            if m[c]["recall"] >= target:
                chosen = float(t)
                break
        out[c] = chosen
    return out


LLM_PROMPT = """You are verifying an automatic event detector on live football commentary (speech-recognition
transcript, may contain errors). The detector claims that a {cls} happens at the line marked >>>.
Decide whether a {cls} ACTUALLY HAPPENS at that moment in the match (for Penalty: a penalty is awarded or
taken now). Answer "no" if the commentators are only watching a replay, recalling an earlier {cls_l},
describing a near miss, arguing it should have been one, or using the word in another sense
(goal kick, goalkeeper, penalty area).

Transcript:
{context}

Reply with exactly one word: yes or no."""


class LLMVerifier:
    def __init__(self, spec: str):
        from .sources.synthetic import make_backend

        parts = spec.split(":")
        kind = parts[1]
        kw = {}
        if len(parts) > 2:
            kw["model"] = ":".join(parts[2:])
        self.backend = make_backend(kind, **kw) if kind == "anthropic" else make_backend("hf", max_new_tokens=4,
                                                                                         temperature=0.0001, **kw)

    def __call__(self, cls: str, context: str) -> float:
        out = self.backend.generate(LLM_PROMPT.format(cls=cls, cls_l=cls.lower(), context=context))
        return 1.0 if re.search(r"\byes\b", out.lower()) else 0.0


def _context(index: dict, game: str, half: int, seg: int, k: int) -> str:
    lines = []
    for j in range(seg - k, seg + k + 1):
        s = index.get((game, half, j))
        if s is not None:
            lines.append((">>> " if j == seg else "    ") + s.text)
    return "\n".join(lines)


def run(cfg: dict, stage1_dir: str, verifier: str, splits=("valid", "test"), target_recall: float = 0.9,
        context_k: int = 4) -> dict:
    pp, ev = cfg["postprocess"], cfg["eval"]
    tol, smooth, nms = ev["tolerance"], pp["smoothing"], pp["nms_window"]
    offsets = class_offsets(cfg)
    st1 = Path(stage1_dir)
    samples = {s: load_split(cfg, s) for s in ("valid",) + tuple(x for x in splits if x != "valid")}
    metas = {s: [{"game": x.game, "half": x.half, "start": x.start, "seg_idx": x.seg_idx} for x in v]
             for s, v in samples.items()}
    gts = {s: load_events(cfg, s) for s in samples}
    p1 = {s: np.load(st1 / f"probs_{s}.npy") for s in samples}
    thr1 = recall_thresholds(metas["valid"], p1["valid"], gts["valid"], tol, smooth, nms, offsets, target_recall)
    cands = {s: detect_events(metas[s], p1[s], thr1, smooth, nms, offsets) for s in samples}

    scores: dict[str, list[float]] = {}
    if verifier.startswith("model:"):
        vdir = Path(verifier.split(":", 1)[1])
        for s in samples:
            p2 = np.load(vdir / f"probs_{s}.npy")
            pos = {(m["game"], m["half"], m["seg_idx"]): i for i, m in enumerate(metas[s])}
            scores[s] = [float(np.sqrt(d.confidence * p2[pos[(d.game, d.half, d.seg_idx)], LABEL2ID[d.cls]]))
                         for d in cands[s]]
    elif verifier.startswith("llm:"):
        llm = LLMVerifier(verifier)
        for s in samples:
            index = {(x.game, x.half, x.seg_idx): x for x in samples[s]}
            scores[s] = [llm(d.cls, _context(index, d.game, d.half, d.seg_idx, context_k)) for d in cands[s]]
    else:
        raise ValueError(f"unknown verifier {verifier!r}")

    def keep(s: str, thr: dict[str, float]) -> list[Detection]:
        return [Detection(d.game, d.half, d.cls, d.time, sc, d.seg_idx)
                for d, sc in zip(cands[s], scores[s]) if sc >= thr[d.cls]]

    thr2 = {}
    for c in EVENT_CLASSES:  # tune verifier threshold per class on valid
        best = (-1.0, 0.5)
        for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
            m = event_metrics([d for d in keep("valid", {k: t for k in EVENT_CLASSES}) if d.cls == c],
                              [g for g in gts["valid"] if g.cls == c], tol)
            best = max(best, (m[c]["f1"], float(t)))
        thr2[c] = best[1]
    res = {"stage1_thresholds": thr1, "stage2_thresholds": thr2, "verifier": verifier, "target_recall": target_recall}
    for s in samples:
        res[s] = {"stage1_candidates": event_metrics(cands[s], gts[s], tol),
                  "two_stage": event_metrics(keep(s, thr2), gts[s], tol)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage1", required=True, help="run dir with probs_<split>.npy")
    ap.add_argument("--verifier", required=True, help="model:<run_dir> | llm:hf:<model> | llm:anthropic[:<model>]")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    res = run(cfg, a.stage1, a.verifier, tuple(a.splits), a.target_recall)
    out = Path(a.out or Path(a.stage1) / "two_stage.json")
    out.write_text(json.dumps(res, indent=1))
    for s in a.splits:
        print(s, "stage1:", json.dumps(res[s]["stage1_candidates"]["overall"]),
              "\n   two-stage:", json.dumps(res[s]["two_stage"]["overall"]))


if __name__ == "__main__":
    main()
