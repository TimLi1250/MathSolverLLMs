#!/usr/bin/env python3
"""Prepare full-trajectory solver data for chat SFT.

Input rows are normalized solver records with:
  problem, solution_steps, final_answer, source

Output rows are chat-format SFT examples. The assistant target is the full
solution trajectory built by concatenating solution_steps, with a final answer
line appended only when final_answer exists.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Provide a complete solution "
    "with all necessary reasoning."
)
DEFAULT_USER_TEMPLATE = "Solve the following problem:\n\n{problem}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, line_no)
            rows.append(row)
    return rows


def validate_row(row: dict[str, Any], line_no: int) -> None:
    for key in ["problem", "solution_steps", "source"]:
        if key not in row:
            raise ValueError(f"Line {line_no}: missing required key {key!r}")
    if not isinstance(row["problem"], str) or not row["problem"].strip():
        raise ValueError(f"Line {line_no}: problem must be a non-empty string")
    if not isinstance(row["solution_steps"], list) or not row["solution_steps"]:
        raise ValueError(f"Line {line_no}: solution_steps must be a non-empty list")
    if not all(isinstance(step, str) and step.strip() for step in row["solution_steps"]):
        raise ValueError(f"Line {line_no}: solution_steps must contain only non-empty strings")
    if "final_answer" in row and row["final_answer"] is not None and not isinstance(row["final_answer"], str):
        raise ValueError(f"Line {line_no}: final_answer must be a string when present")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def build_solution(row: dict[str, Any], final_answer_prefix: str) -> str:
    steps = [normalize_text(step) for step in row["solution_steps"]]
    solution = "\n\n".join(step for step in steps if step).strip()
    final_answer = normalize_text((row.get("final_answer") or "").strip())
    if final_answer:
        solution = f"{solution}\n\n{final_answer_prefix} {final_answer}"
    return solution


def convert_row(
    row: dict[str, Any],
    fallback_index: int,
    system_prompt: str,
    user_template: str,
    final_answer_prefix: str,
) -> dict[str, Any]:
    problem = normalize_text(row["problem"])
    solution = build_solution(row, final_answer_prefix)
    example_index = row.get("example_index", fallback_index)
    source = row["source"]

    converted = {
        "id": f"{source}-{example_index}",
        "example_index": example_index,
        "source": source,
        "problem": problem,
        "solution": solution,
        "final_answer": normalize_text((row.get("final_answer") or "").strip()),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(problem=problem)},
            {"role": "assistant", "content": solution},
        ],
    }
    for optional_key in ["level", "category"]:
        if optional_key in row:
            converted[optional_key] = row[optional_key]
    return converted


def split_rows(
    rows: list[dict[str, Any]],
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["source"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for source_rows in buckets.values():
        shuffled = list(source_rows)
        rng.shuffle(shuffled)
        total = len(shuffled)
        val_count = max(1, math.floor(total * val_size))
        test_count = max(1, math.floor(total * test_size))
        if val_count + test_count >= total:
            val_count = min(1, max(0, total - 1))
            test_count = 0

        val_rows.extend(shuffled[:val_count])
        test_rows.extend(shuffled[val_count : val_count + test_count])
        train_rows.extend(shuffled[val_count + test_count :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict[str, Any]]) -> None:
    sources = Counter(row["source"] for row in rows)
    final_answers = sum(1 for row in rows if row.get("final_answer"))
    print(f"{name}: {len(rows)} examples, final_answer rows={final_answers}, sources={dict(sources)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../normalized_outputs/solver_full_trajectory_dataset.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-name", default="train_sft.jsonl")
    parser.add_argument("--val-name", default="val_sft.jsonl")
    parser.add_argument("--test-name", default="test_sft.jsonl")
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--final-answer-prefix", default="Final Answer:")
    args = parser.parse_args()

    raw_rows = read_jsonl(Path(args.input))
    converted = [
        convert_row(
            row,
            fallback_index=i,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
            final_answer_prefix=args.final_answer_prefix,
        )
        for i, row in enumerate(raw_rows)
    ]

    train_rows, val_rows, test_rows = split_rows(converted, args.val_size, args.test_size, args.seed)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / args.train_name, train_rows)
    write_jsonl(output_dir / args.val_name, val_rows)
    write_jsonl(output_dir / args.test_name, test_rows)

    summarize("train", train_rows)
    summarize("val", val_rows)
    summarize("test", test_rows)
    print(f"Wrote SFT splits to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
