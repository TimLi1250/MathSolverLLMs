# Solver SFT Pipeline

This repo uses staged files for full-trajectory SFT on `HuggingFaceTB/SmolLM2-135M-Instruct`.

## Stage A: Preprocessing

```bash
python prepare_solver_sft_data.py \
  --input solver_full_trajectory_dataset.jsonl \
  --output-dir .
```

Outputs:

- `train_sft.jsonl`
- `val_sft.jsonl`
- `test_sft.jsonl`

Each processed row contains:

- `problem`
- `solution`
- `final_answer`
- `messages`
- source metadata such as `source`, `level`, and `category` when available

The assistant target is the full trajectory formed by concatenating `solution_steps`. Non-empty `final_answer` values are appended as `Final answer: ...` by default.

## Stage B: Tokenizer Formatting Check

```bash
python format_dataset.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --files train_sft.jsonl val_sft.jsonl test_sft.jsonl \
  --max-seq-length 2048
```

This applies the SmolLM2 chat template, validates roles/content, prints token-length stats, and shows rendered examples.

## Stage C: Training

Default full fine-tuning for SmolLM2-135M:

```bash
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  train_solver_sft.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --train-file train_sft.jsonl \
  --val-file val_sft.jsonl \
  --output-dir outputs/smollm2-135m-solver-sft \
  --max-seq-length 2048 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --epochs 5
```

The training script uses assistant-only loss masking. LoRA/QLoRA flags still exist for experimentation, but they are off by default because the 135M model is small enough to fine-tune directly on Colab.

If memory allows, try:

```bash
--max-seq-length 4096
```

If memory is tight, try:

```bash
--max-seq-length 1024 --per-device-train-batch-size 4
```

## Stage D: Inference

After full fine-tuning:

```bash
python run_solver.py \
  --model outputs/smollm2-135m-solver-sft \
  --problem "Compute 55 times 1212 minus 15 times 1212."
```

To test the unfine-tuned base model:

```bash
python run_solver.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --problem "Compute 55 times 1212 minus 15 times 1212."
```

You can also pass `--problem-file` or pipe a problem through stdin.

## Stage E: Evaluation

```bash
python eval_solver.py \
  --model outputs/smollm2-135m-solver-sft \
  --test-file test_sft.jsonl \
  --output-file eval_predictions.jsonl
```

For rows with a non-empty `final_answer`, the evaluator attempts exact-match scoring after extracting `Final answer:` or the last `\boxed{...}` value from the model output.

## Dependencies

```bash
pip install -r requirements-sft.txt
```

The preprocessing script uses only the Python standard library. Formatting, training, inference, and evaluation require the ML dependencies and access to the SmolLM2 model weights.
