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


def test_pretokenize_matches_encode_window_exactly():
    import random

    import torch

    from sed.data.types import Sample
    from sed.model.encoding import LengthGroupedBatchSampler, pretokenize

    words = "goal penalty card the keeper saves what a strike from distance he scores replay".split()
    tok = build_tiny_tokenizer([" ".join(words)] * 5, vocab_size=300)
    rng = random.Random(0)

    def sent(n):
        return " ".join(rng.choice(words) for _ in range(n))

    samples = [Sample(sent(rng.randint(1, 40)), "Goal", "echoes",
                      ctx_before=sent(rng.randint(0, 30)) if rng.random() < 0.8 else "",
                      ctx_after=sent(rng.randint(0, 30)) if rng.random() < 0.8 else "") for _ in range(300)]
    for L in (16, 48, 128):
        assert pretokenize(tok, samples, L, chunk=37) == [
            encode_window(tok, s.ctx_before, s.text, s.ctx_after, L) for s in samples]

    lengths = [rng.randint(5, 100) for _ in range(1000)]
    bs = LengthGroupedBatchSampler(lengths, 32, torch.Generator().manual_seed(0), mega=10)
    batches = list(bs)
    assert len(batches) == len(bs) and sorted(i for b in batches for i in b) == list(range(1000))
