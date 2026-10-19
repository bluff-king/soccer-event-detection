import pytest

from sed.data.types import Segment
from sed.data.windows import make_windows, windows_from_transcript


def test_window_context_and_edges():
    s = [Segment(i, i + 1, f"t{i}") for i in range(4)]
    w = make_windows(s, ["No-Event", "Goal", "No-Event", "Card"], "g", 1, ctx_before=1, ctx_after=1)
    assert len(w) == 4
    assert (w[0].ctx_before, w[0].text, w[0].ctx_after) == ("", "t0", "t1")
    assert (w[1].ctx_before, w[1].text, w[1].ctx_after, w[1].label) == ("t0", "t1", "t2", "Goal")
    assert (w[3].ctx_before, w[3].ctx_after) == ("t2", "")
    w2 = make_windows(s, ["No-Event"] * 4, "g", 1, ctx_before=2, ctx_after=0)
    assert w2[3].ctx_before == "t1 t2" and w2[3].ctx_after == ""


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        make_windows([Segment(0, 1, "a")], [], "g", 1)


def test_windows_from_transcript_dicts_skip_empty():
    w = windows_from_transcript([{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": " "},
                                 {"start": 2, "end": 3, "text": "b"}])
    assert [x.text for x in w] == ["a", "b"] and w[0].ctx_after == "b"
