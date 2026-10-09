from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from ..labels import CLASSES


def class_weights_from_counts(labels: list[str], power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency^power weights, normalised to mean 1 (power 0.5 = inverse sqrt)."""
    c = Counter(labels)
    w = torch.tensor([(1.0 / max(c.get(k, 1), 1)) ** power for k in CLASSES], dtype=torch.float)
    return w / w.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), -1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        return loss.mean()


def build_loss(cfg: dict, train_labels: list[str]) -> torch.nn.Module:
    tr = cfg["train"]
    kind = tr.get("loss", "ce")
    w = None
    if tr.get("class_weights"):
        w = torch.tensor([float(tr["class_weights"][c]) for c in CLASSES])
    elif kind in ("weighted_ce", "focal_weighted"):
        w = class_weights_from_counts(train_labels, tr.get("class_weight_power", 0.5))
    if kind in ("focal", "focal_weighted"):
        return FocalLoss(tr.get("focal_gamma", 2.0), w)
    return torch.nn.CrossEntropyLoss(weight=w)
