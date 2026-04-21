#!/usr/bin/env python3
"""Apply the SmolLM2 chat template and inspect token lengths for processed SFT data."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_transformers() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install with `pip install -r requirements-sft.txt`."
        ) from exc
    return AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_processed_row(row, path, line_no)
            rows.append(row)
    return rows


def validate_processed_row(row: dict[str, Any], path: Path, line_no: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{path}:{line_no}: messages must contain system, user, assistant")
    expected_roles = ["system", "user", "assistant"]
    actual_roles = [message.get("role") for message in messages]
    if actual_roles != expected_roles:
        raise ValueError(f"{path}:{line_no}: expected roles {expected_roles}, got {actual_roles}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{path}:{line_no}: message content must be non-empty")


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * p) - 1))
    return ordered[index]


def summarize_lengths(name: str, lengths: list[int], max_seq_length: int) -> None:
    if not lengths:
        print(f"{name}: no examples")
        return
    truncated = sum(1 for length in lengths if length > max_seq_length)
    print(
        f"{name}: count={len(lengths)}, "
        f"min={min(lengths)}, "
        f"p50={statistics.median(lengths)}, "
        f"p90={percentile(lengths, 0.90)}, "
        f"p95={percentile(lengths, 0.95)}, "
        f"p99={percentile(lengths, 0.99)}, "
        f"max={max(lengths)}, "
        f">{max_seq_length}={truncated}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--files", nargs="+", default=["train_sft.jsonl", "val_sft.jsonl", "test_sft.jsonl"])
    parser.add_argument("--print-examples", type=int, default=1)
    args = parser.parse_args()

    AutoTokenizer = load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for file_name in args.files:
        path = Path(file_name)
        rows = read_jsonl(path)
        full_lengths = []
        prompt_lengths = []

        for row in rows:
            full_text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                row["messages"][:2],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_lengths.append(len(tokenizer(full_text, add_special_tokens=False)["input_ids"]))
            prompt_lengths.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))

        summarize_lengths(str(path), full_lengths, args.max_seq_length)
        summarize_lengths(f"{path} prompts", prompt_lengths, args.max_seq_length)

        for row in rows[: args.print_examples]:
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            print(f"\n--- Rendered example from {path} ({row.get('id')}) ---")
            print(rendered[:2000])
            if len(rendered) > 2000:
                print("... [truncated display]")


if __name__ == "__main__":
    main()
