#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" train_lora_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-file data/train_sft.jsonl \
  --val-file data/val_sft.jsonl \
  --output-dir outputs/qwen2.5-0.5b-solver-lora \
  --max-seq-length 2048 \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --fp16
