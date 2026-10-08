import copy

from sed.reuse import incompatibility

BASE = {"seed": 42, "data": {"delay_min": -5.0, "delay_max": 15.0, "ctx_before": 1},
        "sources": {"caption": {"enabled": False}, "kaggle": {"enabled": False}},
        "model": {"name": "xlm-roberta-base", "max_length": 128},
        "train": {"epochs": 5, "batch_size": 32, "lr": 2e-5, "loss": "ce", "eval_batch_size": 128}}


def test_postprocessing_and_runtime_settings_do_not_block_reuse():
    c = copy.deepcopy(BASE)
    c["postprocess"] = {"smoothing": 3, "nms_window": 45.0}
    c["train"].update(early_stop_patience=1, eval_batch_size=512, num_workers=0)
    c["logging"] = {"wandb": True}
    assert incompatibility(c, BASE) is None


def test_model_changing_settings_block_reuse():
    for path, val in [(("train", "lr"), 3e-5), (("data", "delay_max"), 25.0), (("model", "max_length"), 192),
                      (("sources", "caption"), {"enabled": True}), (("seed",), 1)]:
        c = copy.deepcopy(BASE)
        node = c
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = val
        assert incompatibility(c, BASE) is not None, path
