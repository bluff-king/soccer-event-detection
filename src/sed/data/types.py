"""Small typed containers shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    """One ASR segment. Times are seconds from the start of the half video."""

    start: float
    end: float
    text: str


@dataclass
class Event:
    """A ground-truth event (SoccerNet Labels-v2). `time` is seconds from the start of the half."""

    half: int
    time: float
    label: str  # original SoccerNet label, e.g. "Yellow card"
    cls: str | None  # task class ("Goal"/"Card"/"Penalty") or None for non-highlight events
    team: str = ""


@dataclass
class Sample:
    """One classification sample (a transcript window) with provenance."""

    text: str  # the current segment (or sentence for external sources)
    label: str
    source: str  # "echoes", "caption", "kaggle", "synthetic", "hardneg"
    ctx_before: str = ""  # neighbouring segments (window context)
    ctx_after: str = ""
    game: str = ""
    half: int = 0
    seg_idx: int = -1
    start: float = 0.0
    end: float = 0.0
    split: str = "train"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
