from sed.data.align import label_segments, measure_delays, mentions
from sed.data.types import Event, Segment


def segs(*spans):
    return [Segment(a, b, f"seg {i}") for i, (a, b) in enumerate(spans)]


def test_delay_window_labels_segments_after_event():
    s = segs((0, 2), (2, 4), (10, 12), (14, 16), (30, 32))
    ev = [Event(1, 3.0, "Goal", "Goal")]
    # window [t+0, t+12] = [3, 15] -> segments overlapping it: (2,4), (10,12), (14,16)
    assert label_segments(s, ev, 1, 0, 12) == ["No-Event", "Goal", "Goal", "Goal", "No-Event"]


def test_negative_delay_and_half_filter():
    s = segs((0, 2), (2, 4))
    ev = [Event(1, 5.0, "Yellow card", "Card"), Event(2, 1.0, "Goal", "Goal")]
    # window [t-2, t] = [3, 5] overlaps (2,4) only; [t-4.5, t] = [0.5, 5] overlaps both
    assert label_segments(s, ev, 1, -2, 0) == ["No-Event", "Card"]
    assert label_segments(s, ev, 1, -4.5, 0) == ["Card", "Card"]
    assert label_segments(s, ev, 2, 0, 1) == ["Goal", "No-Event"]


def test_non_highlight_events_ignored():
    s = segs((0, 2))
    assert label_segments(s, [Event(1, 0.5, "Corner", None)], 1, -5, 5) == ["No-Event"]


def test_priority_penalty_over_goal():
    s = segs((10, 12))
    ev = [Event(1, 9.0, "Goal", "Goal"), Event(1, 8.0, "Penalty", "Penalty")]
    assert label_segments(s, ev, 1, 0, 10) == ["Penalty"]
    assert label_segments(s, ev, 1, 0, 10, priority=["Goal", "Penalty", "Card"]) == ["Goal"]


def test_per_class_delays():
    s = segs((0, 2), (50, 52))
    ev = [Event(1, 60.0, "Penalty", "Penalty")]
    assert label_segments(s, ev, 1, 0, 5) == ["No-Event", "No-Event"]
    assert label_segments(s, ev, 1, 0, 5, per_class={"Penalty": (-15, 5)}) == ["No-Event", "Penalty"]


def test_measure_delays_first_mention():
    s = [Segment(0, 2, "nothing"), Segment(12, 14, "and he scores!"), Segment(20, 22, "goal")]
    obs = measure_delays(s, [Event(1, 10.0, "Goal", "Goal")], 1)
    assert obs[0].delay == 2.0
    assert mentions("Card", "he is booked") and not mentions("Goal", "the goalkeeper")
