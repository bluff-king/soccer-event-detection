"""Backbone survey, measured on this machine and on the real data (README section "Chọn backbone").

For each candidate: tokens per window, train throughput / peak VRAM (embeddings frozen or not), inference
throughput, and a short fine-tune (``--steps`` optimizer steps, same seed and data order for every model) scored
on a fixed validation subset. The short fine-tune is an early signal to prune candidates, not a final result.

    python scripts/bench_backbones.py --config configs/local/v2/v2_delays_views.yaml --out docs/backbones/bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from sed.config import load_config, set_seed
from sed.data.prepare import assemble_train, load_split
from sed.labels import LABEL2ID
from sed.metrics import segment_metrics
from sed.model.encoding import Collator, WindowDataset, pretokenize
from sed.model.factory import build_model_and_tokenizer
from sed.model.predict import predict_probs

CANDIDATES = {
    "FacebookAI/xlm-roberta-base": {"amp_dtype": "fp16"},
    "jhu-clsp/mmBERT-base": {"amp_dtype": "bf16"},
    "FacebookAI/roberta-base": {"amp_dtype": "fp16"},
    "vinai/bertweet-base": {"amp_dtype": "fp16"},
    "microsoft/deberta-v3-base": {"amp_dtype": "bf16"},
    "answerdotai/ModernBERT-base": {"amp_dtype": "bf16"},
    "jhu-clsp/ettin-encoder-150m": {"amp_dtype": "bf16"},
}


def throughput(model, tok, samples, bs, freeze, amp, steps=30):
    model.train()
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(not freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    dt = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    dl = torch.utils.data.DataLoader(WindowDataset(samples, tok, 128), batch_size=bs, shuffle=True,
                                     collate_fn=Collator(tok.pad_token_id))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, None
    try:
        for i, b in enumerate(dl):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.time()
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=dt):
                loss = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if i >= 5:
                n += len(b["labels"])
            if i >= 5 + steps:
                break
        torch.cuda.synchronize()
        res = {"samples_per_s": round(n / (time.time() - t0), 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except torch.OutOfMemoryError:
        res = {"samples_per_s": 0.0, "peak_gb": None, "oom": True}
    del opt
    emb.requires_grad_(True)
    gc.collect()
    torch.cuda.empty_cache()
    return res


def short_finetune(cfg, name, amp, train, valid, steps, bs, lr, freeze):
    set_seed(cfg.get("seed", 42))
    c = {**cfg, "model": {**cfg["model"], "name": name, "freeze_embeddings": freeze}}
    model, tok = build_model_and_tokenizer(c)
    model.cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    dt = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 42))
    dl = torch.utils.data.DataLoader(WindowDataset(train, tok, 128), batch_size=bs, shuffle=True, generator=g,
                                     collate_fn=Collator(tok.pad_token_id))
    t0, step, losses = time.time(), 0, []
    while step < steps:
        for b in dl:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=dt):
                loss = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
    train_s = time.time() - t0
    torch.cuda.empty_cache()
    t1 = time.time()
    probs = predict_probs(model, tok, valid, 128, 128, "cuda", True, amp_dtype=amp)
    infer_s = time.time() - t1
    y = np.array([LABEL2ID[s.label] for s in valid])
    m = segment_metrics(y, probs.argmax(1))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"train_s": round(train_s, 1), "final_loss_mean50": round(float(np.mean(losses[-50:])), 4),
            "infer_windows_per_s": round(len(valid) / infer_s, 1),
            "valid_subset": {k: round(m[k], 4) for k in ("accuracy", "macro_f1", "highlight_f1", "highlight_macro_f1")},
            "valid_subset_f1": {c: round(m["per_class"][c]["f1"], 4) for c in m["per_class"]}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local/v2/v2_delays_views.yaml")
    ap.add_argument("--out", default="docs/backbones/bench.json")
    ap.add_argument("--models", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--valid_neg", type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rng = np.random.default_rng(0)
    train = assemble_train(cfg)
    train = [train[i] for i in rng.permutation(len(train))[: a.steps * a.bs + 5000]]
    valid_all = load_split(cfg, "valid")
    pos = [s for s in valid_all if s.label != "No-Event"]
    neg = [valid_all[i] for i in rng.choice([i for i, s in enumerate(valid_all) if s.label == "No-Event"],
                                            a.valid_neg, replace=False)]
    valid = pos + neg
    out_p = Path(a.out)
    results = json.loads(out_p.read_text()) if out_p.exists() else {}
    for name in a.models:
        amp = CANDIDATES.get(name, {}).get("amp_dtype", "fp16")
        print(f"===== {name} ({amp})", flush=True)
        r: dict = {"amp_dtype": amp}
        model, tok = build_model_and_tokenizer({**cfg, "model": {**cfg["model"], "name": name}})
        ids = pretokenize(tok, valid[:5000], 128)
        r["tokens_per_window_mean"] = round(float(np.mean([len(x) for x in ids])), 1)
        r["tokens_per_window_p95"] = int(np.percentile([len(x) for x in ids], 95))
        r["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        r["embedding_params_m"] = round(model.get_input_embeddings().weight.numel() / 1e6, 1)
        model.cuda()
        r["train_frozen_emb"] = throughput(model, tok, train[:3000], a.bs, True, amp)
        r["train_full"] = throughput(model, tok, train[:3000], a.bs, False, amp)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        full_ok = not r["train_full"].get("oom") and r["train_full"]["samples_per_s"] >= 0.5 * \
            r["train_frozen_emb"]["samples_per_s"]
        r["finetune_freeze_embeddings"] = not full_ok  # freeze only when full fine-tuning spills/ooms
        r["finetune"] = short_finetune(cfg, name, amp, train, valid, a.steps, a.bs, a.lr, not full_ok)
        results[name] = r
        print(json.dumps(r), flush=True)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
