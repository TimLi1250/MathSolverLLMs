#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" prepare_full_trajectory_sft.py \
  --input ../normalized_outputs/solver_full_trajectory_dataset.jsonl \
  --output-dir data
