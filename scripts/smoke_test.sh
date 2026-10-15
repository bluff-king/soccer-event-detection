#!/usr/bin/env bash
# One-command CPU smoke test of the whole pipeline (data -> train -> evaluate -> two-stage -> inference API).
# Usage: bash scripts/smoke_test.sh            (downloads public data into data/raw if missing)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
CFG=configs/smoke.yaml
OUT=outputs/smoke

echo "== 1/7 unit tests"
python -m pytest -q

echo "== 2/7 data download (skipped if present)"
if [ ! -d data/raw/sn-echoes/Dataset ] || [ ! -d data/raw/sn-echoes-sushant/summaries ]; then
  python -m sed.data.download --root data/raw
fi

echo "== 3/7 prepare data (3 games per split + all external sources)"
python scripts/prepare_data.py --config $CFG > /dev/null
test -s data/processed_smoke/echoes_valid.jsonl

echo "== 4/7 train + evaluate (tiny offline model, 200 steps)"
python -m sed.train --config $CFG 2>&1 | grep -E "^\[(data|eval|summary)\]"
test -s $OUT/metrics.json && test -s $OUT/best/sed_inference.json

echo "== 5/7 two-stage (stage-1 recall thresholds + model verifier)"
python -m sed.two_stage --config $OUT/config.yaml --stage1 $OUT --verifier model:$OUT --splits valid test
test -s $OUT/two_stage.json

echo "== 6/7 CLI inference"
python -m sed.infer --model $OUT/best --transcript examples/request.json > $OUT/infer_example.json
python -c "import json; d=json.load(open('$OUT/infer_example.json')); print(len(d), 'events'); assert isinstance(d, list)"

echo "== 7/7 HTTP API (uvicorn)"
MODEL_DIR=$OUT/best python -m uvicorn sed.service.app:app --port 8765 > $OUT/uvicorn.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT
for i in $(seq 1 60); do curl -fs localhost:8765/health > /dev/null 2>&1 && break; sleep 1; done
curl -fs -X POST localhost:8765/detect -H 'Content-Type: application/json' -d @examples/request.json \
  | python -c "import json,sys; r=json.load(sys.stdin); print('API ok:', r['n_segments'], 'segments ->', len(r['events']), 'events,', r['latency_ms'], 'ms')"
echo "SMOKE TEST PASSED"
