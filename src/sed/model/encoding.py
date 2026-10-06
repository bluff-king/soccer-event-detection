"""Tokenisation of a window ``before </s> CURRENT </s> after`` that never truncates the current
segment first: the budget left after the current segment is split between the closest context
tokens on each side."""

from __future__ import annotations

import torch

from ..data.types import Sample
from ..labels import LABEL2ID


def combine_ids(tok, b: list[int], cur: list[int], a: list[int], max_length: int) -> list[int]:
    """Assemble ``[CLS] before [SEP] CURRENT [SEP] after [SEP]`` from already tokenised parts."""
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


def encode_window(tok, before: str, text: str, after: str, max_length: int) -> list[int]:
    cur = tok.encode(text, add_special_tokens=False)
    b = tok.encode(before, add_special_tokens=False) if before else []
    a = tok.encode(after, add_special_tokens=False) if after else []
    return combine_ids(tok, b, cur, a, max_length)


def _batch_encode(tok, texts: list[str], chunk: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in texts]
    idx = [i for i, t in enumerate(texts) if t]
    for k in range(0, len(idx), chunk):
        part = idx[k:k + chunk]
        enc = tok([texts[i] for i in part], add_special_tokens=False)["input_ids"]
        for i, e in zip(part, enc):
            out[i] = e
    return out


def pretokenize(tok, samples: list[Sample], max_length: int, chunk: int = 4096) -> list[list[int]]:
    """Token ids for every window, computed once with the (Rust, multi-threaded) batch tokenizer.
    Gives exactly the same ids as ``encode_window`` but keeps the GPU from waiting on Python
    tokenisation in every batch."""
    cur = _batch_encode(tok, [s.text for s in samples], chunk)
    b = _batch_encode(tok, [s.ctx_before for s in samples], chunk)
    a = _batch_encode(tok, [s.ctx_after for s in samples], chunk)
    return [combine_ids(tok, bi, ci, ai, max_length) for bi, ci, ai in zip(b, cur, a)]


class WindowDataset(torch.utils.data.Dataset):
    """Pre-tokenised windows (``ids`` can be passed in to reuse a previous tokenisation)."""

    def __init__(self, samples: list[Sample], tok, max_length: int, ids: list[list[int]] | None = None):
        self.ids = ids if ids is not None else pretokenize(tok, samples, max_length)
        self.labels = [LABEL2ID[s.label] for s in samples]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.ids[i], "label": self.labels[i]}


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Shuffled batches of similar length (less padding -> fewer wasted GPU FLOPs). Indices are
    shuffled, cut into mega-batches of ``mega`` batches, sorted by length inside each mega-batch,
    split into batches, and the batch order is shuffled again."""

    def __init__(self, lengths: list[int], batch_size: int, generator: torch.Generator, mega: int = 50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.generator = generator
        self.mega = mega

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        size = self.batch_size * self.mega
        batches = []
        for k in range(0, len(perm), size):
            chunk = sorted(perm[k:k + size], key=lambda i: -self.lengths[i])
            batches += [chunk[j:j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        for j in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[j]


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
