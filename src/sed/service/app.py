"""FastAPI service.

    MODEL_DIR=outputs/baseline/best uvicorn sed.service.app:app --host 0.0.0.0 --port 8000

POST /detect  {"segments": [{"start": 12.0, "end": 14.5, "text": "...", "half": 1}], "thresholds": {"Goal": 0.6}}
-> {"events": [{"type": "Goal", "timestamp": 790.1, "confidence": 0.93, "segment": {...}}], ...}
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..labels import EVENT_CLASSES


class SegmentIn(BaseModel):
    start: float = Field(..., ge=0, description="segment start (s)")
    end: float = Field(..., ge=0, description="segment end (s)")
    text: str
    half: int | None = Field(None, ge=1, le=2, description="optional half (times restart at 0 in each half)")


class DetectRequest(BaseModel):
    segments: list[SegmentIn] = Field(..., description="ASR segments of one match (or one half), time-ordered")
    thresholds: dict[str, float] | None = Field(None, description="optional per-class threshold override")
    min_confidence: float = 0.0


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class EventOut(BaseModel):
    type: str
    timestamp: float = Field(..., description="estimated event time (s, within the half)")
    confidence: float
    half: int | None = None
    segment: SegmentOut


class DetectResponse(BaseModel):
    events: list[EventOut]
    n_segments: int
    model: str
    latency_ms: float


app = FastAPI(title="Soccer commentary event detection", version="0.1.0")


@lru_cache(maxsize=1)
def get_detector():
    from ..infer import EventDetector

    model_dir = os.environ.get("MODEL_DIR", "/model")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"MODEL_DIR {model_dir!r} not found")
    return EventDetector(model_dir)


@app.get("/health")
def health():
    try:
        det = get_detector()
        return {"status": "ok", "model": str(det.model_dir), "device": det.device, "settings": det.settings}
    except Exception as e:  # pragma: no cover - surfaced to the caller
        return {"status": "error", "detail": str(e)}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if req.thresholds and set(req.thresholds) - set(EVENT_CLASSES):
        raise HTTPException(422, f"threshold keys must be in {EVENT_CLASSES}")
    try:
        det = get_detector()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    t0 = time.time()
    events = det.detect([s.model_dump() for s in req.segments], req.thresholds, req.min_confidence)
    events = [{k: v for k, v in e.items() if v is not None} for e in events]
    return DetectResponse(events=events, n_segments=len(req.segments), model=str(det.model_dir),
                          latency_ms=round((time.time() - t0) * 1000, 1))
