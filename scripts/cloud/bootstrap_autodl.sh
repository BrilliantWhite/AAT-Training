#!/usr/bin/env bash
set -euo pipefail

ROOT="${AAT_ROOT:-/root/autodl-tmp/AAT-Training}"
REPO="${AAT_REPO_DIR:-$ROOT/repo}"

mkdir -p "$ROOT/data" "$ROOT/experiments" "$ROOT/reports" "$ROOT/logs"
cd "$REPO"

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt

echo "AAT environment installed at $REPO/.venv"
