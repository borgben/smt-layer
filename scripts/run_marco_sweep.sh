#!/usr/bin/env bash
# Runs the core baseline first, then MARCO union K=1..8.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export PYTHONPATH="${MARCO_ROOT:-$project_root/../MARCO}:$project_root${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"

"$python_bin" -u -m scripts.marco_sweep \
  --pct 0.5 --pretrain_epochs 30 --epochs 5 \
  --batch_size 1 --lr 1.0 --clip_norm 0.1 --seed 0 \
  --output marco-sweep.csv \
  "$@" \
  2>&1 | tee marco-sweep.log
