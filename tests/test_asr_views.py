import json

import pytest

from sed.data.build import GameData
from sed.data.echoes import EchoesCorpus
from sed.data.types import Event
from sed.sources.asr_views import load_asr_views, view_variants

GAME = "england_epl/2015-2016/2015-08-08 - 19-30 A 1 - 0 B"


def _write(root, variant, half, segs):
    p = root / "Dataset" / variant / GAME / f"{half}_asr.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"segments": {str(i): s for i, s in enumerate(segs)}}))


@pytest.fixture
def corpus_root(tmp_path):
    segs = [[0.0, 5.0, "kick off"], [100.0, 104.0, "he shoots and scores"], [200.0, 204.0, "quiet spell"]]
    for v in ("whisper_v1", "whisper_v2", "whisper_v3", "whisper_v1_en", "whisper_v2_en"):
        _write(tmp_path, v, 1, segs)
    csv = tmp_path / "lang.csv"
    csv.write_text(f"Game/Half,Lang,selected\n{GAME}/1,es,v2\n")
    return tmp_path, csv


def test_list_games_uses_forward_slashes(corpus_root):
    root, csv = corpus_root
    assert EchoesCorpus(root, "auto_en", csv).list_games() == [GAME]


def test_view_variants_exclude_primary_and_filter_language(corpus_root):
    root, csv = corpus_root
    c = EchoesCorpus(root, "auto_en", csv)
    primary, en = view_variants(c, GAME, 1, "en")
    assert primary == "whisper_v2_en"
    assert en == ["whisper_v1_en"]  # Spanish originals are not English text
    _, every = view_variants(c, GAME, 1, "all")
    assert set(every) == {"whisper_v1", "whisper_v2", "whisper_v3", "whisper_v1_en"}


def test_asr_views_labels_and_train_only(corpus_root):
    root, csv = corpus_root
    cfg = {"data": {"echoes_root": str(root), "lang_csv": str(csv), "asr_variant": "auto_en", "delay_min": -5.0,
                    "delay_max": 15.0, "ctx_before": 1, "ctx_after": 1}}
    gd = GameData(GAME, "train", {1: []}, [Event(1, 95.0, "Goal", "Goal")])
    out = load_asr_views(cfg, [gd], "en")
    assert [s.label for s in out] == ["No-Event", "Goal", "No-Event"]
    assert {s.source for s in out} == {"asr_views"} and out[0].meta["variant"] == "whisper_v1_en"
    with pytest.raises(AssertionError):
        load_asr_views(cfg, [GameData(GAME, "test", {1: []}, [])], "en")
