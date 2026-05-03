# Full-Trajectory Solver LoRA SFT

This folder fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on the normalized solver trajectories.

The data prep step reads:

```text
../normalized_outputs/solver_full_trajectory_dataset.jsonl
```

It writes chat SFT splits under:

```text
data/train_sft.jsonl
data/val_sft.jsonl
data/test_sft.jsonl
```

For each example, the assistant target is:

1. `solution_steps` concatenated with blank lines.
2. `Final Answer: ...` appended only when `final_answer` exists.

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./prepare_data.sh
./train_lora.sh
```

For Google Colab T4 training, open `colab_lora_sft.ipynb` in this folder and run it top to bottom.
After training, open `colab_show_download_results.ipynb` to inspect metrics, run sample generations, and download or copy the adapter results.

The LoRA adapter output goes to:

```text
outputs/qwen2.5-0.5b-solver-lora
```

If you need Hugging Face authentication in your environment, log in with:

```bash
.venv/bin/huggingface-cli login
```

This local setup has already generated the SFT splits under `data/`.
