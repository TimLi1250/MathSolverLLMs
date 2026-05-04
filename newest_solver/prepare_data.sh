#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" prepare_math_only_sft.py \
  --input ../normalized_outputs/solver_full_trajectory_dataset.jsonl \
  --output-dir data \
  --limit 2000 \
  --skip-normalized-output
