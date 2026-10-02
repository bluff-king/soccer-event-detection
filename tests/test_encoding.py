from sed.model.factory import build_tiny_tokenizer
from sed.model.encoding import encode_window


def test_encode_keeps_current_segment_and_budget():
    tok = build_tiny_tokenizer(["goal for the home side what a strike from distance"] * 5, vocab_size=200)
    before = " ".join(["what a strike"] * 30)
    after = " ".join(["from distance"] * 30)
    ids = encode_window(tok, before, "goal for the home side", after, 32)
    assert len(ids) <= 32
    cur = tok.encode("goal for the home side", add_special_tokens=False)
    s = " ".join(map(str, ids))
    assert " ".join(map(str, cur)) in s
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    short = encode_window(tok, "", "goal", "", 32)
    assert short == [tok.cls_token_id] + tok.encode("goal", add_special_tokens=False) + [tok.sep_token_id]
