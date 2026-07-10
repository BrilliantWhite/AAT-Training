#!/usr/bin/env bash
set -euo pipefail

ROOT="${AAT_ROOT:-/root/autodl-tmp/AAT-Training}"
REPO="${AAT_REPO_DIR:-$ROOT/repo}"
CURRENT_PID_FILE="${AAT_CURRENT_PID_FILE:-$ROOT/logs/EXP-FROZEN-V1-RESNET18-002.pid}"
CURRENT_EXPERIMENT="${AAT_CURRENT_EXPERIMENT:-EXP-FROZEN-V1-RESNET18-002}"

cd "$REPO"
source .venv/bin/activate
source /etc/network_turbo 2>/dev/null || true

manifest_status() {
  python - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "run_manifest.json"
print(json.loads(path.read_text())["status"] if path.is_file() else "missing")
PY
}

current_pid="$(cat "$CURRENT_PID_FILE")"
while kill -0 "$current_pid" 2>/dev/null; do
  sleep 60
done
if [[ "$(manifest_status "$ROOT/experiments/$CURRENT_EXPERIMENT")" != "complete" ]]; then
  echo "STOP: $CURRENT_EXPERIMENT did not complete" >&2
  exit 1
fi

run_model() {
  local config="$1"
  local experiment_id="$2"
  local log="$ROOT/logs/$experiment_id.log"
  test ! -e "$ROOT/experiments/$experiment_id"
  python -u scripts/train_cnn_baselines.py \
    --config "$config" \
    --inputs-dir "$ROOT/data/frozen_v1_inputs_v1" \
    --folds "$ROOT/data/frozen_v1_folds_v1/folds.csv" \
    --dataset-manifest "$ROOT/data/frozen_v1/manifest.json" \
    --fold-manifest "$ROOT/data/frozen_v1_folds_v1/manifest.json" \
    --experiments-root "$ROOT/experiments" \
    --experiment-id "$experiment_id" \
    --dataset-version frozen_v1 \
    --fold-version folds_v1 \
    --seed 20260710 \
    --device cuda 2>&1 | tee "$log"
  if [[ "$(manifest_status "$ROOT/experiments/$experiment_id")" != "complete" ]]; then
    echo "STOP: $experiment_id exited without a complete manifest" >&2
    exit 1
  fi
}

run_model configs/training/efficientnet_b0_v1.yaml EXP-FROZEN-V1-EFFICIENTNET-B0-001
run_model configs/training/resnet50_proposal_v1.yaml EXP-FROZEN-V1-RESNET50-PROPOSAL-001
run_model configs/training/inception_v3_proposal_v1.yaml EXP-FROZEN-V1-INCEPTION-V3-PROPOSAL-001

date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/formal_cnn_sequence.complete"
echo "PASS: all four formal CNN experiments completed"

