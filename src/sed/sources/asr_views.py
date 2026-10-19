"""Extra real ASR views of the TRAIN games: SoccerNet-Echoes ships up to 5 transcripts per half (Whisper v1/v2/v3
originals + v1/v2 English translations for non-English commentary) but the main dataset uses one of them.
The other transcripts are real in-domain ASR of the same commentary with the same labels, so they multiply the
real event windows (Penalty included) without any domain shift. Valid/test never use them.

``mode``: ``en`` keeps English text only (English originals + translations), ``all`` also keeps the native-language
originals (useful for a multilingual backbone).
"""

from __future__ import annotations

from ..data.align import label_segments
from ..data.echoes import VARIANTS, EchoesCorpus, read_asr_json, resolve_variant
from ..data.types import Sample
from ..data.windows import make_windows


def view_variants(corpus: EchoesCorpus, game: str, half: int, mode: str = "en") -> tuple[str | None, list[str]]:
    """(variant used by the main dataset, other existing variants to add as views)."""
    exists = [v for v in VARIANTS if (corpus.dataset / v / game / f"{half}_asr.json").exists()]
    primary = next((v for v in resolve_variant(game, half, corpus.variant, corpus.lang_table) if v in exists), None)
    lang = corpus.lang(game, half)
    out = []
    for v in exists:
        if v == primary:
            continue
        english = v.endswith("_en") or lang == "en"
        if mode == "en" and not english:
            continue
        out.append(v)
    return primary, out


def load_asr_views(cfg: dict, games, mode: str = "en") -> list[Sample]:
    """``games``: GameData of the train split (events already loaded)."""
    from ..data.build import per_class_delays

    d = cfg["data"]
    corpus = EchoesCorpus(d["echoes_root"], d.get("asr_variant", "auto_en"), d.get("lang_csv"))
    out: list[Sample] = []
    for gd in games:
        assert gd.split == "train", "ASR views are train-only"
        for h in gd.halves:
            _, views = view_variants(corpus, gd.game, h, mode)
            for v in views:
                segs = read_asr_json(corpus.dataset / v / gd.game / f"{h}_asr.json")
                if not segs:
                    continue
                labels = label_segments(segs, gd.events, h, d["delay_min"], d["delay_max"],
                                        per_class=per_class_delays(cfg))
                ws = make_windows(segs, labels, gd.game, h, d["ctx_before"], d["ctx_after"], "train", "asr_views")
                for w in ws:
                    w.meta = {"variant": v, "lang": corpus.lang(gd.game, h)}
                out += ws
    return out
