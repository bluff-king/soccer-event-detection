"""Inference on one match transcript: list of segments -> list of events.

    python -m sed.infer --model outputs/baseline/best --transcript examples/transcript_example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.windows import windows_from_transcript
from .labels import EVENT_CLASSES
from .model.factory import load_checkpoint
from .model.predict import get_device, predict_probs
from .postprocess import detect_events

DEFAULT_SETTINGS = {"thresholds": {c: 0.5 for c in EVENT_CLASSES}, "offsets": {}, "smoothing": 1,
                    "nms_window": 30.0, "ctx_before": 1, "ctx_after": 1, "max_length": 128}


class EventDetector:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device or get_device()
        self.model, self.tok = load_checkpoint(self.model_dir, self.device)
        p = self.model_dir / "sed_inference.json"
        self.settings = {**DEFAULT_SETTINGS, **(json.loads(p.read_text()) if p.exists() else {})}

    def detect(self, segments: list[dict], thresholds: dict[str, float] | None = None,
               min_confidence: float = 0.0) -> list[dict]:
        """``segments``: dicts with start, end, text and optional ``half`` (1/2). Halves are processed
        separately (context, smoothing and NMS never cross a half; times restart in each half)."""
        st = self.settings
        windows, halves = [], []
        for h in sorted({int(s.get("half") or 0) for s in segments}):
            ws = windows_from_transcript([s for s in segments if int(s.get("half") or 0) == h],
                                         st["ctx_before"], st["ctx_after"])
            windows += ws
            halves += [h] * len(ws)
        if not windows:
            return []
        probs = predict_probs(self.model, self.tok, windows, st["max_length"], 64, self.device)
        meta = [{"game": "match", "half": h, "start": w.start, "seg_idx": i}
                for i, (w, h) in enumerate(zip(windows, halves))]
        thr = {**st["thresholds"], **(thresholds or {})}
        dets = detect_events(meta, probs, thr, st["smoothing"], st["nms_window"], st["offsets"])
        out = []
        for d in dets:
            if d.confidence < min_confidence:
                continue
            w = windows[d.seg_idx]
            e = {"type": d.cls, "timestamp": round(max(0.0, d.time), 2), "confidence": round(d.confidence, 4),
                 "segment": {"start": w.start, "end": w.end, "text": w.text}}
            if d.half:
                e["half"] = d.half
            out.append(e)
        return sorted(out, key=lambda e: (e.get("half", 0), e["timestamp"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcript", required=True,
                    help="JSON: {segments: [{start,end,text[,half]}]} or a SoccerNet-Echoes *_asr.json")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.transcript).read_text())
    segs = data["segments"]
    if isinstance(segs, dict):  # SoccerNet-Echoes format {idx: [start, end, text]}
        segs = [{"start": v[0], "end": v[1], "text": v[2]} for v in segs.values()]
    print(json.dumps(EventDetector(a.model).detect(segs), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
