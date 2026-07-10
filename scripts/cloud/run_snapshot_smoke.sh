#!/usr/bin/env bash
set -euo pipefail

ROOT="${AAT_ROOT:-/root/autodl-tmp/AAT-Training}"
REPO="${AAT_REPO_DIR:-$ROOT/repo}"
DATASET_MANIFEST="${AAT_DATASET_MANIFEST:-$ROOT/data/frozen_v1/manifest.json}"
FOLD_MANIFEST="${AAT_FOLD_MANIFEST:-$ROOT/data/frozen_v1_folds_v1/manifest.json}"
FOLDS="${AAT_FOLDS:-$ROOT/data/frozen_v1_folds_v1/folds.csv}"

cd "$REPO"
source .venv/bin/activate

python scripts/cloud/verify_cloud_environment.py \
  --dataset-manifest "$DATASET_MANIFEST" \
  --fold-manifest "$FOLD_MANIFEST" \
  --folds "$FOLDS"

python scripts/cloud/gpu_smoke.py

echo "PASS: cloud environment and two-batch GPU smoke completed"

