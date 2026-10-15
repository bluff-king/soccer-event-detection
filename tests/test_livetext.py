from sed.sources.livetext import clean, label_match


def _r(minute, typ, text):
    return {"Minute": minute, "Type": typ, "Comment": text, "_order": 0}


def test_penalty_window_and_hard_negatives():
    rows = [
        _r(10, "comment", "big shout for a penalty but the referee waves play on"),  # no kick near -> negative
        _r(20, "comment", "he drives towards the edge of the penalty area"),         # not a penalty at all
        _r(58, "highlight", "penalty to arsenal! the referee points to the spot"),    # 2 min before the kick
        _r(59, "var", "var checks the penalty decision"),
        _r(60, "PSG", "goal! saka sends the keeper the wrong way from the spot"),
        _r(61, "G", "goal! another one"),
        _r(70, "YC", "booked for a late challenge"),
        _r(75, "comment", "that penalty earlier changed the game"),                 # after the kick -> negative
    ]
    assert label_match(rows, 2.0) == ["No-Event", "No-Event", "Penalty", "Penalty", "Penalty", "Goal", "Card",
                                      "No-Event"]


def test_penalty_saved_highlight_is_a_kick():
    rows = [_r(30, "highlight", "penalty to city, foul in the box"), _r(31, "highlight", "the penalty is saved!")]
    assert label_match(rows) == ["Penalty", "Penalty"]


def test_clean_is_asr_like():
    assert clean("GOOOAAALLL!!! salah�s penalty") == "goal. salah's penalty"
    assert clean("what a save!!!") == "what a save."
