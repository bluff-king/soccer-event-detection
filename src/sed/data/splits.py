"""Match-level train/valid/test split.

Default: the official SoccerNet-v2 split (300/100/100 games) from the ``SoccerNet`` pip package
(MIT), which is also the split used by SoccerNet-Caption -> captions of valid/test games can be
excluded consistently. Fallback (package missing or ``method: hash``): deterministic hash of the
game path, so the split never depends on iteration order.
"""

from __future__ import annotations

import hashlib


def official_splits() -> dict[str, str] | None:
    try:
        from SoccerNet.utils import getListGames
    except Exception:  # pragma: no cover - depends on optional package
        return None
    out = {}
    for split in ("train", "valid", "test"):
        for g in getListGames(split):
            out[g] = split
    return out


def hash_split(game: str, valid_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> str:
    h = int(hashlib.sha1(f"{seed}:{game}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + valid_frac:
        return "valid"
    return "train"


def assign_splits(games: list[str], method: str = "official", valid_frac: float = 0.15,
                  test_frac: float = 0.15, seed: int = 42) -> dict[str, str]:
    """Return game -> split. Games outside the official lists (e.g. the 50 challenge games, which
    have no public labels) get split ``"unlabelled"`` under the official method."""
    if method == "official":
        table = official_splits()
        if table is not None:
            return {g: table.get(g, "unlabelled") for g in games}
        method = "hash"
    if method == "hash":
        return {g: hash_split(g, valid_frac, test_frac, seed) for g in games}
    raise ValueError(f"unknown split method {method!r}")


def check_disjoint(games_by_split: dict[str, set[str]]) -> None:
    """Raise if any game appears in more than one split (guards against context leakage)."""
    names = list(games_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = games_by_split[a] & games_by_split[b]
            if common:
                raise AssertionError(f"{len(common)} game(s) in both {a} and {b}: {sorted(common)[:3]}")
