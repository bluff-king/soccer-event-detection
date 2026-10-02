"""Training loop (plain PyTorch, fp16 autocast on GPU, linear warmup/decay, per-epoch validation and
best-checkpoint selection), followed by the full evaluation.

    python -m sed.train --config configs/base.yaml [overrides...]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config, set_seed
from .data.prepare import assemble_train, load_split
from .evaluate import evaluate_checkpoint, summary_row
from .labels import LABEL2ID
from .metrics import segment_metrics
from .model.encoding import Collator, WindowDataset
from .model.factory import build_model_and_tokenizer, load_checkpoint
from .model.losses import build_loss
from .model.predict import get_device, predict_probs


def make_sampler(cfg: dict, labels: list[str]):
    kind = cfg["train"].get("sampler", "none")
    if kind == "none":
        return None
    c = Counter(labels)
    power = 1.0 if kind == "balanced" else 0.5  # "sqrt": milder re-balancing
    w = [1.0 / c[x] ** power for x in labels]
    return torch.utils.data.WeightedRandomSampler(w, num_samples=len(labels), replacement=True)


def train(cfg: dict) -> dict:
    set_seed(cfg.get("seed", 42))
    tr = cfg["train"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = get_device()

    train_samples = assemble_train(cfg)
    valid_samples = load_split(cfg, "valid")
    if tr.get("max_eval_samples"):  # subsample validation for fast per-epoch selection (smoke tests)
        rng = np.random.default_rng(cfg.get("seed", 42))
        idx = rng.choice(len(valid_samples), size=min(tr["max_eval_samples"], len(valid_samples)), replace=False)
        valid_samples = [valid_samples[i] for i in sorted(idx)]
    labels = [s.label for s in train_samples]
    print(f"[data] train={len(train_samples)} {dict(Counter(labels))} "
          f"sources={dict(Counter(s.source for s in train_samples))} valid={len(valid_samples)}", flush=True)

    model, tok = build_model_and_tokenizer(
        cfg, [" ".join([s.ctx_before, s.text, s.ctx_after]) for s in train_samples[:50000]])
    model.to(device)
    ds = WindowDataset(train_samples, tok, cfg["model"]["max_length"])
    sampler = make_sampler(cfg, labels)
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))
    dl = torch.utils.data.DataLoader(ds, batch_size=tr["batch_size"], shuffle=sampler is None, sampler=sampler,
                                     collate_fn=Collator(tok.pad_token_id), num_workers=tr.get("num_workers", 0),
                                     generator=g, drop_last=False)
    loss_fn = build_loss(cfg, labels).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": tr.get("weight_decay", 0.01)},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(tr["lr"]))
    steps_per_epoch = math.ceil(len(dl) / tr.get("grad_accum", 1))
    total = steps_per_epoch * tr["epochs"]
    if tr.get("max_steps"):
        total = min(total, tr["max_steps"])
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(opt, int(total * tr.get("warmup_ratio", 0.06)), total)
    use_amp = tr.get("fp16", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_every = tr.get("eval_every") or steps_per_epoch
    select = tr.get("select_metric", "highlight_f1")
    best, history, step = -1.0, [], 0
    t0 = time.time()

    def run_eval(tag: str):
        nonlocal best
        probs = predict_probs(model, tok, valid_samples, cfg["model"]["max_length"], tr["eval_batch_size"], device,
                              tr.get("fp16", True))
        y = np.array([LABEL2ID[s.label] for s in valid_samples])
        m = segment_metrics(y, probs.argmax(1))
        rec = {"step": step, "tag": tag, "elapsed_s": round(time.time() - t0, 1),
               **{k: m[k] for k in ("accuracy", "macro_f1", "highlight_f1", "highlight_precision", "highlight_recall")}}
        history.append(rec)
        print(f"[eval] {json.dumps(rec)}", flush=True)
        if m[select] > best:
            best = m[select]
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        model.train()

    model.train()
    done = False
    for epoch in range(tr["epochs"]):
        for i, b in enumerate(dl):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = loss_fn(logits.float(), b["labels"]) / tr.get("grad_accum", 1)
            scaler.scale(loss).backward()
            if (i + 1) % tr.get("grad_accum", 1) == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0:
                    print(f"[train] epoch={epoch} step={step}/{total} loss={loss.item() * tr.get('grad_accum', 1):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                if step % eval_every == 0:
                    run_eval(f"epoch{epoch}")
                if step >= total:
                    done = True
                    break
        if done:
            break
    if not history or history[-1]["step"] != step:
        run_eval("final")
    train_time = time.time() - t0
    (out / "history.json").write_text(json.dumps(history, indent=1))

    model, tok = load_checkpoint(out / "best", device)
    res = evaluate_checkpoint(cfg, model, tok, out, ckpt_dir=out / "best")
    res["train_time_s"] = train_time
    res["n_train"] = len(train_samples)
    res["train_class_counts"] = dict(Counter(labels))
    res["train_source_counts"] = dict(Counter(s.source for s in train_samples))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    row = summary_row(cfg.get("run_name", out.name), res, "valid")
    row.update({f"test_{k}": v for k, v in summary_row("", res, "test").items() if k != "run"})
    row["train_time_min"] = round(train_time / 60, 1)
    (out / "summary.json").write_text(json.dumps(row, indent=1))
    print("[summary]", json.dumps(row), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
