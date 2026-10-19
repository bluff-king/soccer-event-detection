"""Why do some backbones use far more memory than their size suggests, or produce NaN losses?

For each model: attention implementation, peak memory of one forward+backward at batch 32 for fp32 / fp16 / bf16
autocast and for each available attention implementation, and whether the loss / gradients are finite.

    python scripts/diagnose_backbones.py --out docs/backbones/diagnose.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from sed.config import load_config
from sed.data.prepare import load_split
from sed.labels import ID2LABEL, LABEL2ID
from sed.model.encoding import Collator, WindowDataset


def one_step(model, batch, dtype):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        out.loss.float().backward()
        finite_grad = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        logits = out.logits.float()
        return {"loss": round(out.loss.item(), 4), "loss_finite": bool(torch.isfinite(out.loss).item()),
                "logits_absmax": round(logits.abs().max().item(), 2), "grads_finite": finite_grad,
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except torch.OutOfMemoryError:
        return {"oom": True}
    finally:
        model.zero_grad(set_to_none=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["answerdotai/ModernBERT-base", "jhu-clsp/ettin-encoder-150m",
                                                     "microsoft/deberta-v3-base", "FacebookAI/roberta-base"])
    ap.add_argument("--out", default="docs/backbones/diagnose.json")
    a = ap.parse_args(argv)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_config("configs/local/v2/v2_delays.yaml")
    samples = load_split(cfg, "valid")[:2000:20][:32]
    res = {}
    for name in a.models:
        tok = AutoTokenizer.from_pretrained(name)
        b = next(iter(torch.utils.data.DataLoader(WindowDataset(samples, tok, 128), batch_size=32,
                                                  collate_fn=Collator(tok.pad_token_id))))
        b = {k: v.cuda() for k, v in b.items()}
        r = {"seq_len": int(b["input_ids"].shape[1])}
        for attn in ("sdpa", "eager"):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, attn_implementation=attn).cuda()
            except Exception as e:  # implementation not supported by this architecture
                r[attn] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                continue
            r[attn] = {"config_attn": getattr(model.config, "_attn_implementation", None),
                       **{k: one_step(model, b, dt) for k, dt in (("fp32", None), ("fp16", torch.float16),
                                                                  ("bf16", torch.bfloat16))}}
            del model
            gc.collect()
            torch.cuda.empty_cache()
        res[name] = r
        print(name, json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
