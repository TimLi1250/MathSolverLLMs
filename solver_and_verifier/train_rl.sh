#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" train_rl_solver.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --solver-adapter ../newest_solver/outputs/qwen2.5-0.5b-math-only-lora \
  --verifier-dir ../verifier/modernbert_joint_verifier_best \
  --train-file ../newest_solver/data/train_sft.jsonl \
  --output-dir outputs/qwen2.5-0.5b-math-verifier-rl-lora \
  --num-updates 200 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.95 \
  --kl-coef 0.02 \
  --reward-batch-size 4 \
  --dtype fp16 \
  --device-map auto \
  --reference-device-map auto
