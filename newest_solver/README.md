# MATH-Only Full-Trajectory Solver LoRA

This folder trains a fresh LoRA adapter from the base
`Qwen/Qwen2.5-0.5B-Instruct` model on MATH-only full-solution trajectories.
Each assistant target is built from all available `solution_steps` and ends with
a `Final Answer: <answer>` line.

Colab flow:

1. Copy or sync this `newest_solver` folder and `normalized_outputs/solver_full_trajectory_dataset.jsonl` into Google Drive under `MyDrive/Final Project/`.
2. Open `colab_math_only_sft.ipynb`.
3. Run cells top to bottom.

The data prep reads up to 2,000 unique MATH examples with non-empty
`final_answer` from `../normalized_outputs/solver_full_trajectory_dataset.jsonl`.
If the source file has fewer eligible rows, the prep script prints a warning
and uses all eligible rows. It writes the SFT splits:

- `data/train_sft.jsonl`
- `data/val_sft.jsonl`
- `data/test_sft.jsonl`

Training runs for 3 epochs and writes the adapter to:

- `outputs/qwen2.5-0.5b-math-only-lora`
