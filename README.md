# MathSolverLLMs

Full-trajectory SFT pipeline for training a `HuggingFaceTB/SmolLM2-135M-Instruct` math solver on `solver_full_trajectory_dataset.jsonl`.

Start with:

```bash
python prepare_solver_sft_data.py --input solver_full_trajectory_dataset.jsonl --output-dir .
```

Then inspect formatting/token lengths:

```bash
python format_dataset.py
```

Train:

```bash
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no train_solver_sft.py
```

See `SFT_README.md` for the full staged workflow.
