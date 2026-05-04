# Verifier-Guided Math Solver

This repository contains a small end-to-end pipeline for training a mathematical
problem solver with verifier-guided reinforcement learning. The solver starts
from `Qwen/Qwen2.5-0.5B-Instruct`, is supervised fine-tuned on MATH-style full
solution trajectories, and is then RL-tuned with a frozen ModernBERT verifier as
the reward model.

The main experiment compares three solver variants:

| Model | Accuracy | Numeric parse coverage | Avg. percent error | Length ratio | ROUGE-L | BLEU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 base | 0.30 | 0.529 | 153.26 | 6.72 | 0.243 | 0.053 |
| MATH-only SFT | 0.25 | 0.824 | 65.97 | 1.48 | 0.407 | 0.196 |
| SFT + verifier RL | 0.35 | 0.850 | 48.50 | 1.35 | 0.430 | 0.215 |

These results are preliminary because the evaluation set is small, but they show
the intended trend: supervised fine-tuning improves formatting and solution
style, while verifier-guided RL improves answer correctness while preserving the
more concise SFT format.

## Repository Layout

```text
.
├── method_section.tex
├── dataset_solver.ipynb
├── normalized_outputs/
│   └── solver_full_trajectory_dataset.jsonl
├── newest_solver/
│   ├── prepare_math_only_sft.py
│   ├── train_lora_sft.py
│   ├── prepare_data.sh
│   ├── train_lora.sh
│   ├── colab_math_only_sft.ipynb
│   └── data/
│       ├── train_sft.jsonl
│       ├── val_sft.jsonl
│       └── test_sft.jsonl
├── verifier/
│   ├── prm800k_preprocessing.ipynb
│   ├── modernbert_joint_verifier.ipynb
│   ├── solution_verification.jsonl
│   └── next_step_verification.jsonl
└── solver_and_verifier/
    ├── reward_model.py
    ├── score_with_verifier.py
    ├── train_rl_solver.py
    ├── train_rl.sh
    └── colab_solver_verifier_rl.ipynb
```

## Data

Run `dataset_solver.ipynb` first to build the normalized solver dataset. The
notebook loads `EleutherAI/hendrycks_math`, extracts 2,000 unique MATH problems
with full solution trajectories and final answers, and writes:

```text
normalized_outputs/solver_full_trajectory_dataset.jsonl
```

This JSONL file contains normalized solver trajectories with fields such as
`problem`, `solution_steps`, `final_answer`, `source`, `level`, and `category`.

`newest_solver/colab_math_only_sft.ipynb` converts that normalized JSONL into
chat-format SFT splits. The checked-in split has:

| File | Rows |
| --- | ---: |
| `newest_solver/data/train_sft.jsonl` | 1,144 |
| `newest_solver/data/val_sft.jsonl` | 63 |
| `newest_solver/data/test_sft.jsonl` | 63 |

`verifier/solution_verification.jsonl` contains 905 full-solution verification
examples. `verifier/next_step_verification.jsonl` contains 26,256 next-step
verification examples.

## Setup

The notebooks are the primary pipeline and are set up for Google Colab. The
shell scripts can be run locally in a GPU environment if the required models and
checkpoints are available.

Install dependencies for solver SFT:

```bash
cd newest_solver
pip install -r requirements.txt
```

Install dependencies for RL:

```bash
cd solver_and_verifier
pip install -r requirements.txt
```

## Pipeline

### 1. Build the normalized solver dataset

Open and run `dataset_solver.ipynb` from the repository root. It downloads the
MATH data, normalizes each problem and solution trajectory, extracts final
answers, and writes:

```text
normalized_outputs/solver_full_trajectory_dataset.jsonl
```

### 2. Run the MATH-only solver notebook

Open and run:

```text
newest_solver/colab_math_only_sft.ipynb
```

This notebook prepares the chat-format SFT splits from
`normalized_outputs/solver_full_trajectory_dataset.jsonl` and trains the
MATH-only LoRA solver. It writes the SFT data to:

```text
newest_solver/data/
```

and the trained solver adapter to:

```text
newest_solver/outputs/qwen2.5-0.5b-math-only-lora/
```

For local runs, the same preparation and training steps are available as:

```bash
cd newest_solver
bash prepare_data.sh
bash train_lora.sh
```

### 3. Run the verifier notebook

Open and run:

```text
verifier/modernbert_joint_verifier.ipynb
```

This trains the ModernBERT verifier on full-solution and next-step verification
examples. The RL stage expects the trained verifier checkpoint at:

```text
verifier/modernbert_joint_verifier_best/
```

`verifier/prm800k_preprocessing.ipynb` is the preprocessing notebook used to
rebuild the verifier JSONL files if needed.

### 4. Run the solver-and-verifier RL notebook

Open and run:

```text
solver_and_verifier/colab_solver_verifier_rl.ipynb
```

This notebook loads the MATH-only solver adapter and the trained verifier,
samples solver outputs, scores them with the verifier, and trains the final
verifier-guided RL solver adapter. RL outputs are written to:

```text
solver_and_verifier/outputs/qwen2.5-0.5b-math-verifier-rl-lora/
```

For local runs, the equivalent command is:

```bash
cd solver_and_verifier
bash train_rl.sh
```

## Optional Checks

### Smoke-test the verifier reward

From `solver_and_verifier/`:

```bash
python score_with_verifier.py \
  --verifier-dir ../verifier/modernbert_joint_verifier_best \
  --input-file ../newest_solver/data/test_sft.jsonl \
  --limit 8
```

This loads the trained verifier, formats MATH solutions with the same prompt
template used during verifier training, and reports `P(correct_or_valid)`.

## Key Implementation Details

- Solver base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Solver tuning method: LoRA on attention and MLP projection modules
- SFT target format: full reference reasoning followed by `Final Answer: <answer>`
- Verifier base model: `answerdotai/ModernBERT-large`
- Verifier labels: `incorrect_or_invalid` and `correct_or_valid`
- Reward: verifier probability for the positive class, `P(correct_or_valid)`
- RL default generation: temperature `0.7`, top-p `0.95`, max `512` new tokens
- RL regularization: sampled-token KL surrogate against the frozen SFT policy

## Notes

Large model checkpoints and generated `outputs/` directories are not part of the
current checked-in tree. Train the SFT solver and verifier first, or pass custom
checkpoint paths to the scripts with `--solver-adapter` and `--verifier-dir`.

For the Colab flow, keep the same relative directory layout under
`MyDrive/Final Project/` so the notebooks and scripts can resolve paths without
modification.
