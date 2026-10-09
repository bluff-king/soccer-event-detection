import json
import sys
import types
from pathlib import Path

from sed.tracking import Tracker, run_display_name

RES = Path(__file__).resolve().parents[1] / "docs" / "results" / "cpu_tiny"


class _Run:
    url = "https://wandb.example/run"

    def __init__(self):
        self.logged, self.summary_data = [], {}
        self.summary = types.SimpleNamespace(update=self.summary_data.update)

    def log(self, data, step=None):
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


def _fake_wandb(monkeypatch):
    mod = types.ModuleType("wandb")
    mod.inits = []

    def init(**kw):
        mod.inits.append(kw)
        mod.run = _Run()
        return mod.run

    mod.init = init
    mod.Table = lambda columns, data: ("table", columns, data)
    mod.Image = lambda path: ("image", path)
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def _cfg(**logging):
    return {"output_dir": "outputs/x", "run_name": "data_caption", "model": {"name": "xlm-roberta-base", "max_length": 128},
            "train": {"epochs": 5}, "logging": logging}


def test_disabled_or_missing_key_is_noop(monkeypatch):
    _fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    for t in (Tracker(_cfg(wandb=False)), Tracker(_cfg(wandb=True))):
        assert t.run is None
        t.log({"a": 1}), t.summary({"b": 2}), t.finish()  # no errors


def test_logs_results_and_name(monkeypatch):
    wb = _fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    cfg = _cfg(wandb=True, project="football-highlight", entity="team", group="g1", name_prefix="colab")
    assert run_display_name(cfg) == "colab_data_caption_xlmr-base_len128_ep5"
    t = Tracker(cfg)
    assert wb.inits[0]["entity"] == "team" and wb.inits[0]["group"] == "g1" and wb.inits[0]["project"] == "football-highlight"
    res = json.loads((RES / "metrics.json").read_text())
    row = json.loads((RES / "summary.json").read_text())
    t.log({"train/loss": 0.5}, step=50)
    t.log_results(res, RES, row)
    s = wb.run.summary_data
    assert s["valid/event/overall/f1"] == res["valid"]["event"]["overall"]["f1"]
    assert s["test/event/Penalty/f1"] == res["test"]["event"]["Penalty"]["f1"]
    assert s["summary/evt_f1"] == row["evt_f1"] and "thresholds/Goal" in s and "valid/fp/replay_recall" in s
    logged = {k for _, d in wb.run.logged for k in d}
    assert {"train/loss", "results_table", "pr_curves_valid", "confusion_valid"} <= logged
    t.finish()
    assert t.run is None
