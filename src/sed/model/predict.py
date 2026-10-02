from __future__ import annotations

import numpy as np
import torch

from ..data.types import Sample
from .encoding import Collator, WindowDataset


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_probs(model, tok, samples: list[Sample], max_length: int, batch_size: int = 128,
                  device: str | None = None, fp16: bool = True, num_workers: int = 0) -> np.ndarray:
    device = device or get_device()
    model.eval().to(device)
    # sort by length for fast batched inference, then restore order
    order = np.argsort([len(s.text) + len(s.ctx_before) + len(s.ctx_after) for s in samples])
    ds = WindowDataset([samples[i] for i in order], tok, max_length)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=Collator(tok.pad_token_id),
                                     num_workers=num_workers)
    out = []
    use_amp = fp16 and device == "cuda"
    for b in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, model.config.num_labels), dtype=np.float32)
    res = np.empty_like(probs)
    res[order] = probs
    return res
