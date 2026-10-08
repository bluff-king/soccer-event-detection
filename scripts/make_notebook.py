"""Generate notebooks/train_colab.ipynb (kept as code so the notebook stays reviewable in diffs).

    python scripts/make_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Commentary event detection: train & evaluate on Colab (T4)

Runs end to end: clone repo, install, download public data, data survey, prepare datasets (Echoes +
external sources), keyword reference, **baseline `xlm-roberta-base`**, data ablations, improvements,
two-stage verification, results table, and saving to Google Drive.

**Runtime → Change runtime type → T4 GPU.** Estimated times (T4, fp16) are given in each section.
Every run writes `outputs/<run>/summary.json`; finished runs are skipped when re-run, so a disconnected
session can resume (outputs are synced to Drive if `USE_DRIVE=True`).
""")

code("""
# ---- settings -------------------------------------------------------------
REPO_URL = "https://github.com/bluff-king/soccer-event-detection"
BRANCH = "claude/sweet-einstein-slyhy7"  # switch to "main" once the PR is merged
USE_DRIVE = True           # persist outputs/ to Google Drive
RUN_SET = "core"           # "core": baseline + 5 data ablations + 3 improvements | "full": + 5 more (measured on T4:
                           # ~45-90 min per run with 5 epochs, roughly half with early stopping)
OFFICIAL_LABELS = False    # also download official Labels-v2 via the SoccerNet package (mirror is used otherwise)
SYNTHETIC_BACKEND = "template"  # "template" (instant) | "hf" (Qwen2.5-1.5B-Instruct on the T4, ~40 min) | "anthropic"
FAST_MODE = False          # True: length-grouped batches of 64 + early stopping (best epoch is usually 0-1). Changes the
                           # training batches, so results go to a separate folder (outputs_fast) and are only
                           # comparable with other FAST_MODE runs.
EARLY_STOP = True          # stop a run when validation stops improving (best epoch is usually 0-1); the kept
                           # checkpoint is the same as with all 5 epochs, so results stay comparable
EARLY_STOP_PATIENCE = 1    # epochs without improvement before stopping: 1 = fastest, 2 = safer against noise
USE_WANDB = True           # log runs to Weights & Biases
WANDB_PROJECT = "football-highlight"
WANDB_ENTITY = "vubkk67-hanoi-university-of-science-and-technology"
WANDB_SECRET = "kgat"      # name of the Colab Secret (key icon, left bar) whose VALUE is the W&B API key

_over = []
if FAST_MODE:
    _over += ["train.group_by_length=true", "train.batch_size=64", "train.lr=3e-5"]
if EARLY_STOP or FAST_MODE:
    _over += [f"train.early_stop_patience={int(EARLY_STOP_PATIENCE)}"]
if USE_WANDB:
    import time as _time
    _over += ["logging.wandb=true", f"logging.project={WANDB_PROJECT}", f"logging.entity={WANDB_ENTITY}",
              f"logging.group=colab_{_time.strftime('%Y%m%d_%H%M')}", "logging.name_prefix=colab"]
EXTRA_SET = ("--set " + " ".join(_over)) if _over else ""
# xlm-roberta-large keeps its own batch size / lr
LARGE_SET = ("--set " + " ".join(o for o in _over if not o.startswith(("train.batch_size", "train.lr")))) if _over else ""
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, subprocess
%cd /content
# absolute paths: re-running this cell from inside the repo must update it, not clone a second copy into it
if not os.path.exists("/content/soccer-event-detection/.git"):
    !git clone -b $BRANCH $REPO_URL /content/soccer-event-detection
else:  # runtime kept from an earlier run: get the latest code (data/ and outputs/ are untouched)
    !git -C /content/soccer-event-detection fetch -q origin $BRANCH && git -C /content/soccer-event-detection checkout -q $BRANCH && git -C /content/soccer-event-detection reset -q --hard origin/$BRANCH
%cd /content/soccer-event-detection
!git log --oneline -3
!pip -q install -e . "SoccerNet>=0.1.60"
if USE_WANDB:
    !pip -q install -U "wandb>=0.22"  # new-format keys (wandb_v1_...) need a recent client
    from google.colab import userdata
    try:
        _key = userdata.get(WANDB_SECRET)
    except Exception as e:
        _key = None
        print(f"Cannot read Colab Secret {WANDB_SECRET!r} ({e}); enable 'Notebook access' for it. W&B logging is off.")
    if _key:
        import wandb
        try:  # verify the key against the W&B server instead of guessing its format
            wandb.login(key=_key.strip(), relogin=True, verify=True)
            os.environ["WANDB_API_KEY"] = _key.strip()  # training subprocesses read the key from the environment
            print("W&B login ok (key from Colab Secret", repr(WANDB_SECRET) + ")")
        except Exception as e:
            print(f"W&B login FAILED with the value of secret {WANDB_SECRET!r}: {e}")
            print("Copy the key again from https://wandb.ai/authorize (check Name/Value are not swapped). "
                  "Training continues without W&B logging.")
""")

code("""
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_OUT = "/content/drive/MyDrive/soccer-event-detection/" + ("outputs_fast" if FAST_MODE else "outputs")
    os.makedirs(DRIVE_OUT, exist_ok=True)
    # keep outputs/ on Drive so finished runs survive disconnects
    if not os.path.islink("outputs"):
        !rm -rf outputs && ln -s $DRIVE_OUT outputs
!ls -la outputs || mkdir -p outputs
if USE_WANDB and os.environ.get("WANDB_API_KEY"):
    # upload runs finished in earlier sessions (once; marked with outputs/<run>/.wandb_synced)
    !python scripts/wandb_backfill.py outputs/* --set logging.project=$WANDB_PROJECT logging.entity=$WANDB_ENTITY logging.name_prefix=colab
""")

md("## 1. Data download (~2 min) and survey")
code("""
args = "--official" if OFFICIAL_LABELS else ""
!python -m sed.data.download --root data/raw $args
!python scripts/survey_data.py --config configs/base.yaml --out outputs/survey > /dev/null
from IPython.display import Markdown, Image, display
display(Markdown(open("outputs/survey/DATA_SURVEY.md").read().replace("figures/", "outputs/survey/figures/")))
display(Image("outputs/survey/figures/delay_profile.png"))
""")

md("""
## 2. Prepare datasets (~2 min with template synthesis)
Validation/test = real SoccerNet-Echoes ASR only (official match split). External sources go to train only;
leakage into valid/test games is asserted.
""")
code("""
over = []
if SYNTHETIC_BACKEND != "template":
    over.append(f"sources.synthetic.backend={SYNTHETIC_BACKEND}")
!python scripts/prepare_data.py --config configs/base.yaml {" ".join(over)} | tail -60
""")

md("## 3. Sanity check: unit tests + end-to-end smoke test (~2 min)")
code("""
!python -m pytest -q
!bash scripts/smoke_test.sh 2>&1 | tail -8
""")

md("## 4. Keyword-rule reference (seconds)")
code("""
!python scripts/keyword_baseline.py --config configs/base.yaml | head -30
""")

md("""
## 5. Baseline: xlm-roberta-base, 3-segment window, 5 epochs, fp16 (~20 min)
Reference reported for the original project: accuracy ≈ 72.4 %, F1-highlight ≈ 0.67 on validation.
Their evaluation set was most likely class-balanced; compare with `bal_acc` / `bal_highlight_f1`
(natural-distribution numbers are much harsher because ~97 % of windows are No-Event).
""")
code("""
!python scripts/run_experiments.py configs/base.yaml $EXTRA_SET
""")

md("## 6. Data ablation: Echoes only vs. + each source vs. + all (~2.5 h)")
code("""
# "Echoes only" = the baseline run of section 5 (configs/ablation/data_echoes.yaml is the same config)
DATA_RUNS = ["configs/ablation/data_caption.yaml",
             "configs/ablation/data_synthetic.yaml", "configs/ablation/data_hardneg.yaml",
             "configs/ablation/data_kaggle.yaml", "configs/ablation/data_all.yaml"]
!python scripts/run_experiments.py {" ".join(DATA_RUNS)} $EXTRA_SET
""")

md("""
## 7. Improvements vs. baseline (one change at a time, Echoes only)
`imp_class_delays` changes the labels, so compare it on **event-level** metrics only.
""")
code("""
IMP_RUNS = ["configs/ablation/imp_weighted_ce.yaml", "configs/ablation/imp_class_delays.yaml",
            "configs/ablation/imp_smoothing.yaml"]
if RUN_SET == "full":
    IMP_RUNS += ["configs/ablation/imp_focal.yaml", "configs/ablation/imp_context5.yaml",
                 "configs/ablation/imp_mdeberta.yaml", "configs/ablation/imp_all_negatives.yaml"]
!python scripts/run_experiments.py {" ".join(IMP_RUNS)} $EXTRA_SET
if RUN_SET == "full":  # large model keeps its own batch size / lr; FAST_MODE only groups by length here
    !python scripts/run_experiments.py configs/ablation/imp_xlmr_large.yaml $LARGE_SET
""")

md("""
## 8. Two-stage: high-recall stage 1 + verifier
Model verifier needs `imp_xlmr_large` (RUN_SET="full"); otherwise the context-5 or weighted-CE run is used.
The LLM verifier (Qwen 1.5B on the T4) is slow: it is run on validation only by default.
""")
code("""
import os
verifier = next((d for d in ["outputs/imp_xlmr_large", "outputs/imp_context5", "outputs/imp_weighted_ce"]
                 if os.path.exists(d + "/probs_valid.npy")), None)
if verifier:
    !python -m sed.two_stage --config outputs/baseline/config.yaml --stage1 outputs/baseline --verifier model:$verifier --out outputs/baseline/two_stage_model.json
# LLM verifier (optional, ~1-2 s per candidate):
# !python -m sed.two_stage --config outputs/baseline/config.yaml --stage1 outputs/baseline --verifier llm:hf:Qwen/Qwen2.5-1.5B-Instruct --splits valid --out outputs/baseline/two_stage_llm.json
""")

md("## 9. Final candidate (edit `configs/final.yaml` after looking at the ablations) (~45 min)")
code("""
!python scripts/run_experiments.py configs/final.yaml $EXTRA_SET
""")

md("## 10. Results")
code("""
import json, glob, pandas as pd
runs = ["outputs/keyword_baseline", "outputs/baseline"] + sorted(set(glob.glob("outputs/data_*") + glob.glob("outputs/imp_*"))) + ["outputs/final"]
rows = [json.load(open(r + "/summary.json")) for r in runs if os.path.exists(r + "/summary.json")]
df = pd.DataFrame(rows).set_index("run")
cols = ["seg_acc", "seg_macro_f1", "seg_highlight_f1", "bal_acc", "bal_highlight_f1",
        "evt_precision", "evt_recall", "evt_f1", "evt_f1_Goal", "evt_f1_Card", "evt_f1_Penalty",
        "test_evt_precision", "test_evt_recall", "test_evt_f1", "train_time_min"]
display(df[[c for c in cols if c in df]].round(3))
print(df[[c for c in cols if c in df]].round(3).to_markdown())
""")
code("""
from IPython.display import Image, display
for run in ["outputs/baseline", "outputs/final"]:
    if os.path.exists(run + "/metrics.json"):
        m = json.load(open(run + "/metrics.json"))
        print(run, "thresholds:", m["thresholds"])
        print(" FP groups (valid):", {k: v["count"] for k, v in m["valid"]["fp_diagnosis"]["groups"].items()})
        print(" event F1 argmax/no NMS -> tuned+NMS:", round(m["valid"]["event_argmax_no_nms"]["overall"]["f1"], 3),
              "->", round(m["valid"]["event"]["overall"]["f1"], 3))
        display(Image(run + "/pr_curves_valid.png"), Image(run + "/confusion_valid.png"))
""")

md("## 11. Save artifacts (results table + best checkpoint) and try the API")
code("""
!cp -r outputs/survey/figures docs_figures_colab 2>/dev/null || true
!cd outputs && zip -qr ../results_bundle.zip */summary.json */metrics.json */history.json */config.yaml */*.png results.md 2>/dev/null; ls -lh ../results_bundle.zip 2>/dev/null || ls -lh results_bundle.zip
best = "outputs/final/best" if os.path.exists("outputs/final/best") else "outputs/baseline/best"
!python -m sed.infer --model $best --transcript examples/request.json | head -30
print("Download results_bundle.zip and", best, "(model for the Docker service).")
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "train_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out)
