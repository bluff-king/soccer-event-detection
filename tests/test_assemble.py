import pytest

from sed.data.build import save_jsonl, subsample_negatives
from sed.data.prepare import assemble_train
from sed.data.types import Sample


def _cfg(tmp_path, **sources):
    return {"seed": 0, "data": {"processed_dir": str(tmp_path), "train_neg_ratio": None},
            "sources": {k: {"enabled": v} for k, v in sources.items()}}


def _write_splits(tmp_path):
    save_jsonl([Sample("a", "Goal", "echoes", game="g_train"), Sample("b", "No-Event", "echoes", game="g_train")],
               tmp_path / "echoes_train.jsonl")
    save_jsonl([Sample("c", "No-Event", "echoes", game="g_valid", split="valid")], tmp_path / "echoes_valid.jsonl")
    save_jsonl([Sample("d", "No-Event", "echoes", game="g_test", split="test")], tmp_path / "echoes_test.jsonl")


def test_external_sources_only_enter_train(tmp_path):
    _write_splits(tmp_path)
    save_jsonl([Sample("cap", "Card", "caption", game="g_train")], tmp_path / "caption.jsonl")
    tr = assemble_train(_cfg(tmp_path, caption=True))
    assert [s.source for s in tr] == ["echoes", "echoes", "caption"]
    assert len(assemble_train(_cfg(tmp_path, caption=False))) == 2


def test_leakage_of_heldout_game_is_rejected(tmp_path):
    _write_splits(tmp_path)
    save_jsonl([Sample("cap", "Card", "caption", game="g_valid")], tmp_path / "caption.jsonl")
    with pytest.raises(AssertionError):
        assemble_train(_cfg(tmp_path, caption=True))


def test_subsample_negatives_keeps_positives_and_keyword_negatives():
    s = [Sample("goal!", "Goal", "echoes", game="g", seg_idx=0)]
    s += [Sample("what a goal that was last week", "No-Event", "echoes", game="g", seg_idx=1)]
    s += [Sample(f"pass {i}", "No-Event", "echoes", game="g", seg_idx=2 + i) for i in range(50)]
    out = subsample_negatives(s, ratio=3.0, seed=0)
    assert sum(x.label == "Goal" for x in out) == 1
    assert any(x.seg_idx == 1 for x in out)  # keyword negative kept
    assert sum(x.label == "No-Event" for x in out) == 3
