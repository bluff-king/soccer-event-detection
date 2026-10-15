import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from sed.model.factory import build_model_and_tokenizer

    d = tmp_path_factory.mktemp("model")
    cfg = {"model": {"name": "tiny-local", "max_length": 64}}
    model, tok = build_model_and_tokenizer(cfg, ["what a goal he scores", "yellow card for the defender"] * 10)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    # thresholds of 0 so the random model emits events and the response schema is exercised
    (d / "sed_inference.json").write_text(json.dumps({"thresholds": {"Goal": 0.0, "Card": 0.0, "Penalty": 0.0},
                                                      "offsets": {"Goal": 5.0}, "smoothing": 1, "nms_window": 30.0,
                                                      "ctx_before": 1, "ctx_after": 1, "max_length": 64}))
    return d


def test_detect_endpoint(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from sed.service import app as app_mod

    app_mod.get_detector.cache_clear()
    client = TestClient(app_mod.app)
    assert client.get("/health").json()["status"] == "ok"
    segs = [{"start": float(i * 2), "end": float(i * 2 + 2), "text": t}
            for i, t in enumerate(["he shoots", "and he scores", "what a goal", "the replay now"] * 30)]
    r = client.post("/detect", json={"segments": segs})
    assert r.status_code == 200
    body = r.json()
    assert body["n_segments"] == len(segs) and body["events"]
    e = body["events"][0]
    assert {"type", "timestamp", "confidence", "segment"} <= set(e) and e["type"] in ("Goal", "Card", "Penalty")
    # NMS: events of one class are at least nms_window apart (compare segment starts: timestamps are
    # clipped at 0 after subtracting the class offset)
    goals = sorted(x["segment"]["start"] for x in body["events"] if x["type"] == "Goal")
    assert all(b - a >= 30 for a, b in zip(goals, goals[1:]))
    assert client.post("/detect", json={"segments": segs, "thresholds": {"Foo": 1}}).status_code == 422
    # two halves with restarting times are processed separately
    two = [dict(s, half=1) for s in segs] + [dict(s, half=2) for s in segs]
    b2 = client.post("/detect", json={"segments": two}).json()
    assert {e["half"] for e in b2["events"]} == {1, 2}
    for h in (1, 2):
        g = sorted(e["segment"]["start"] for e in b2["events"] if e["type"] == "Goal" and e["half"] == h)
        assert all(b - a >= 30 for a, b in zip(g, g[1:]))
    assert client.post("/detect", json={"segments": []}).json()["events"] == []
