"""Label space and mapping from SoccerNet Labels-v2 classes to the 4-class task."""

from __future__ import annotations

CLASSES = ["No-Event", "Goal", "Card", "Penalty"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
NO_EVENT = 0
EVENT_CLASSES = CLASSES[1:]  # the "highlight" group

# SoccerNet-v2 (17 classes) -> task classes. Anything not listed is No-Event.
SOCCERNET_TO_TASK = {
    "Goal": "Goal",
    "Yellow card": "Card",
    "Red card": "Card",
    "Yellow->red card": "Card",
    "Penalty": "Penalty",
}

# When one segment is covered by several events, the higher-priority class wins.
# Penalty first because it is the rarest class and a penalty is usually followed by a goal
# ~30-90 s later (the goal then labels its own later segments).
DEFAULT_PRIORITY = ["Penalty", "Goal", "Card"]


def map_soccernet_label(label: str) -> str | None:
    """Return the task class for a SoccerNet-v2 label, or None if it is not a highlight class."""
    return SOCCERNET_TO_TASK.get(label)
