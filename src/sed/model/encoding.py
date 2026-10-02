"""Tokenisation of a window ``before </s> CURRENT </s> after`` that never truncates the current
segment first: the budget left after the current segment is split between the closest context
tokens on each side."""

from __future__ import annotations

import torch

from ..data.types import Sample
from ..labels import LABEL2ID


def encode_window(tok, before: str, text: str, after: str, max_length: int) -> list[int]:
    cur = tok.encode(text, add_special_tokens=False)
    b = tok.encode(before, add_special_tokens=False) if before else []
    a = tok.encode(after, add_special_tokens=False) if after else []
    sep = [tok.sep_token_id]
    cls = [tok.cls_token_id]
    n_special = 2 + (1 if b else 0) + (1 if a else 0)  # cls + final sep + inner seps
    budget = max_length - n_special
    cur = cur[:budget]
    rest = budget - len(cur)
    # split remaining budget, giving any unused share to the other side
    nb = min(len(b), rest // 2)
    na = min(len(a), rest - nb)
    nb = min(len(b), rest - na)
    ids = cls
    if b:
        ids = ids + b[len(b) - nb:] + sep
    ids = ids + cur
    if a:
        ids = ids + sep + a[:na]
    return ids + sep


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], tok, max_length: int):
        self.samples = samples
        self.tok = tok
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        return {"input_ids": encode_window(self.tok, s.ctx_before, s.text, s.ctx_after, self.max_length),
                "label": LABEL2ID[s.label]}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
            mask[i, : len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)}
