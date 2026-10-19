import random

from sed.data.soccernet_labels import dedupe_events, parse_mirror_text
from sed.sources.caption import label_mirror_captions
from sed.sources.kaggle_events import kaggle_label, mask_leaks, overlapping_kaggle_games, soccernet_game_key
from sed.sources.synthetic import SynthConfig, generate_hard_negatives, generate_synthetic
from sed.sources.text_utils import asr_noise, strip_scores

MIRROR = """
    A 1 - 0 B

First Half:

00:00 Event: Kick-off by B
13:10 Event: Goal by A
13:10 Event: Goal by A
13:15 Caption: GOAL 1:0! Smith (A) scores a header.
20:00 Caption: Smith (A) shoots from the edge of the penalty area, wide.
Second Half:
05:00 Event: Yellow card by B
05:20 Caption: The referee shows a yellow card to Jones (B).
44:00 Event: Penalty by A
"""


def test_parse_mirror_and_dedupe():
    ev = parse_mirror_text(MIRROR)
    labs = [(e.half, e.time, e.label, e.cls, e.team) for e in ev]
    assert (1, 790.0, "Goal", "Goal", "A") in labs
    assert (2, 300.0, "Yellow card", "Card", "B") in labs
    assert (1, 0.0, "Kick-off", None, "B") in labs
    kept, dropped = dedupe_events(ev, 5)
    assert dropped == 1 and len(kept) == len(ev) - 1


def test_caption_alignment_labels():
    caps = label_mirror_captions(MIRROR)
    assert [c[3] for c in caps] == ["Goal", "No-Event", "Card"]


def test_kaggle_masking_removes_leaks():
    t = "Goal!  Everton 1, Leicester City 3. Shinji Okazaki (Leicester City) left footed shot from the centre of the box."
    m = mask_leaks(t)
    assert "Goal" not in m and "1," not in m and "3." not in m and "Okazaki" in m
    m2 = mask_leaks("Booking      Graziano Pelle (Southampton) is shown the yellow card for a bad foul.")
    assert "yellow" not in m2.lower() and "card" not in m2.lower() and "Booking" not in m2
    m3 = mask_leaks("Penalty Napoli. Gonzalo Higuain draws a foul in the penalty area.")
    assert "penalty" not in m3.lower()
    assert kaggle_label({"is_goal": "1", "event_type": "1"}) == "Goal"
    assert kaggle_label({"is_goal": "0", "event_type": "6"}) == "Card"
    assert kaggle_label({"is_goal": "0", "event_type": "11"}) == "Penalty"
    assert kaggle_label({"is_goal": "0", "event_type": "2"}) == "No-Event"


def test_kaggle_overlap_detection():
    g = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    assert soccernet_game_key(g)[0].isoformat() == "2015-02-21"
    ginf = [{"id_odsp": "x", "date": "2015-02-21", "ht": "Chelsea", "at": "Burnley"},
            {"id_odsp": "y", "date": "2015-02-21", "ht": "Arsenal", "at": "Burnley"}]
    assert overlapping_kaggle_games(ginf, [g]) == {"x"}


def test_strip_scores_and_noise():
    assert "2-1" not in strip_scores("it is 2-1 now")
    rng = random.Random(0)
    out = asr_noise("The referee shows a yellow card to Jones.", rng)
    assert isinstance(out, str) and out


def test_synthetic_template_generation_marks_source():
    syn = generate_synthetic({"Penalty": ["x"]}, SynthConfig(targets={"Penalty": 7, "Goal": 3}))
    assert len(syn) == 10 and {s.source for s in syn} == {"synthetic"}
    assert sum(s.label == "Penalty" for s in syn) == 7
    hn = generate_hard_negatives(SynthConfig(hardneg_per_category=2))
    assert hn and all(s.label == "No-Event" and s.source == "hardneg" and s.meta["category"] for s in hn)
