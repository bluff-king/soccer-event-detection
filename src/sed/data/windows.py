"""Sliding-window sample generation: one sample per ASR segment, with neighbouring segments as
context (default 1 before + current + 1 after = 3 segments, as in the reference baseline)."""

from __future__ import annotations

from .types import Sample, Segment


def make_windows(
    segments: list[Segment],
    labels: list[str],
    game: str,
    half: int,
    ctx_before: int = 1,
    ctx_after: int = 1,
    split: str = "train",
    source: str = "echoes",
) -> list[Sample]:
    if len(segments) != len(labels):
        raise ValueError("segments and labels must have the same length")
    out = []
    n = len(segments)
    for i, s in enumerate(segments):
        before = " ".join(x.text for x in segments[max(0, i - ctx_before):i])
        after = " ".join(x.text for x in segments[i + 1:min(n, i + 1 + ctx_after)])
        out.append(Sample(
            text=s.text, label=labels[i], source=source, ctx_before=before, ctx_after=after,
            game=game, half=half, seg_idx=i, start=s.start, end=s.end, split=split,
        ))
    return out


def windows_from_transcript(segments: list[dict] | list[Segment], ctx_before: int = 1,
                            ctx_after: int = 1) -> list[Sample]:
    """Unlabelled windows for inference. Accepts dicts with start/end/text or Segment objects."""
    segs = [s if isinstance(s, Segment) else Segment(float(s["start"]), float(s["end"]), str(s["text"]))
            for s in segments]
    segs = [s for s in segs if s.text.strip()]
    return make_windows(segs, ["No-Event"] * len(segs), game="", half=0,
                        ctx_before=ctx_before, ctx_after=ctx_after, split="infer")
