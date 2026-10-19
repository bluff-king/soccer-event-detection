"""Minimal client for the detection API.

    python examples/client.py --url http://localhost:8000 --transcript examples/request.json
"""

import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--transcript", default="examples/request.json")
    a = ap.parse_args()
    body = json.load(open(a.transcript, encoding="utf-8"))
    req = urllib.request.Request(a.url.rstrip("/") + "/detect", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.load(r)
    for e in res["events"]:
        m, s = divmod(e["timestamp"], 60)
        print(f"{int(m):02d}:{s:04.1f}  {e['type']:<8} conf={e['confidence']:.2f}  \"{e['segment']['text']}\"")
    print(f"{len(res['events'])} events from {res['n_segments']} segments in {res['latency_ms']} ms")


if __name__ == "__main__":
    main()
