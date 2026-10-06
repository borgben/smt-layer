#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root so module imports and data paths resolve.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Prefer the project's virtual environment when available.
python_bin=python
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
fi

# Baseline
"$python_bin" -u -m scripts.mnist_sudoku \
  --pct 0.5 --pretrain_epochs 30 --epochs 5 \
  --batch_size 1 --lr 1.0 --clip_norm 0.1 \
  --measure-singleton-cache-hits --conflict-gradient core --seed 0 --no-early-stop \
  2>&1 | tee core-seed0.log

# Update all singleton-conflicting bits
"$python_bin" -u -m scripts.mnist_sudoku \
  --pct 0.5 --pretrain_epochs 30 --epochs 5 \
  --batch_size 1 --lr 1.0 --clip_norm 0.1 \
  --measure-singleton-cache-hits --conflict-gradient all_singletons --seed 0 --no-early-stop \
  2>&1 | tee singletons-seed0.log
