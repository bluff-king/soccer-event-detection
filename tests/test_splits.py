import pytest

from sed.data.splits import assign_splits, check_disjoint, hash_split, official_splits


def test_hash_split_deterministic_and_disjoint():
    games = [f"league/season/game{i}" for i in range(500)]
    a = assign_splits(games, "hash")
    b = assign_splits(list(reversed(games)), "hash")
    assert a == b
    by = {s: {g for g, x in a.items() if x == s} for s in ("train", "valid", "test")}
    check_disjoint(by)
    assert sum(map(len, by.values())) == 500
    assert 0.6 < len(by["train"]) / 500 < 0.8
    assert hash_split("x", seed=1) in ("train", "valid", "test")


def test_official_split_disjoint():
    table = official_splits()
    if table is None:
        pytest.skip("SoccerNet package not installed")
    by = {s: {g for g, x in table.items() if x == s} for s in ("train", "valid", "test")}
    check_disjoint(by)
    assert (len(by["train"]), len(by["valid"]), len(by["test"])) == (300, 100, 100)


def test_check_disjoint_detects_leak():
    with pytest.raises(AssertionError):
        check_disjoint({"train": {"a", "b"}, "valid": {"b"}})
