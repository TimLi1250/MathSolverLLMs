# Solver + Verifier RL

This folder fine-tunes the MATH-only solver from `newest_solver` with the
ModernBERT verifier from `verifier` as a reward model.

The verifier is used exactly as in `verifier/modernbert_joint_verifier_colab_prefix_truncation.ipynb`:

- load it with `AutoTokenizer` and `AutoModelForSequenceClassification`
- format verifier inputs with the `Instruction`, `Task`, `Problem`, and `Solution` fields
- apply `softmax(logits)[:, 1]`
- treat that probability as `P(correct_or_valid)` and use it as the reward

## Files

- `reward_model.py` wraps the verifier model and preserves the verifier notebook's input format.
- `score_with_verifier.py` smoke-tests the verifier on known SFT rows.
- `train_rl_solver.py` runs lightweight policy-gradient RL with a KL penalty against the MATH-only SFT adapter.
- `train_rl.sh` is the default Colab/local launch command.
- `colab_solver_verifier_rl.ipynb` is a small Colab runner for the same flow.

## Expected Layout

From the repo root:

```text
newest_solver/
  data/train_sft.jsonl
  outputs/qwen2.5-0.5b-math-only-lora/
verifier/
  modernbert_joint_verifier_best/
solver_and_verifier/
```

On Google Drive, use the same relative layout under `MyDrive/Final Project/`.
If your verifier checkpoint is stored elsewhere, pass that path with
`--verifier-dir`, for example `/content/drive/MyDrive/modernbert_joint_verifier_best`.

## Install

```bash
cd solver_and_verifier
pip install -r requirements.txt
```

ModernBERT needs a recent `transformers`; the requirement is set to `>=4.48.0`.

## Smoke-Test The Reward Model

```bash
python score_with_verifier.py \
  --verifier-dir ../verifier/modernbert_joint_verifier_best \
  --input-file ../newest_solver/data/test_sft.jsonl \
  --limit 8
```

This scores gold MATH solutions from the SFT data. It verifies that the model
loads, the prompt format matches the verifier notebook, and class `1` is being
used as the reward.

## RL Fine-Tuning

```bash
bash train_rl.sh
```

The trainer defaults to full-solution reward. It samples a solution from the
current solver, asks the verifier for `P(correct_or_valid)`, then updates the
solver LoRA adapter with a policy-gradient loss plus a KL-style penalty against
the original MATH-only adapter.

Useful knobs:

```bash
python train_rl_solver.py \
  --num-updates 300 \
  --train-batch-size 2 \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --kl-coef 0.02 \
  --next-step-reward-weight 0.0
```

Set `--next-step-reward-weight` above `0.0` only if you want additional shaping
from the verifier's next-step task. The default is solution-level reward because
that exactly matches generated solver outputs.

Outputs are written to:

```text
solver_and_verifier/outputs/qwen2.5-0.5b-math-verifier-rl-lora/
```
