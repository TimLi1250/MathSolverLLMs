#!/usr/bin/env python3
"""Prepare MATH-only full-solution trajectory data for solver SFT.

Input rows are normalized solver records with:
  problem, solution_steps, final_answer, source

This script filters to source == "math" and non-empty final_answer, writes the
filtered normalized JSONL, and creates chat-format train/val/test SFT splits.
Each assistant target contains the full solution trajectory from solution_steps
followed by a final-answer line.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Provide a complete solution "
    "with all necessary reasoning, and end with a line of the form "
    "'Final Answer: <answer>'."
)
DEFAULT_USER_TEMPLATE = "Solve the following problem:\n\n{problem}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_normalized_row(row, line_no)
            rows.append(row)
    return rows


def validate_normalized_row(row: dict[str, Any], line_no: int) -> None:
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


def filter_math_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    for row in rows:
        if row.get("source") != "math":
            continue
        if not normalize_text((row.get("final_answer") or "").strip()):
            continue
        problem = normalize_text(row["problem"])
        if problem in seen_problems:
            continue
        seen_problems.add(problem)
        new_row = dict(row)
        new_row["problem"] = problem
        new_row["solution_steps"] = [normalize_text(step) for step in row["solution_steps"] if normalize_text(step)]
        new_row["final_answer"] = normalize_text((row.get("final_answer") or "").strip())
        new_row["source"] = "math"
        new_row["example_index"] = len(filtered)
        filtered.append(new_row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def build_solution(row: dict[str, Any], final_answer_prefix: str) -> str:
    solution = "\n\n".join(row["solution_steps"]).strip()
    final_answer = normalize_text((row.get("final_answer") or "").strip())
    return f"{solution}\n\n{final_answer_prefix} {final_answer}".strip()


def convert_row(
    row: dict[str, Any],
    system_prompt: str,
    user_template: str,
    final_answer_prefix: str,
) -> dict[str, Any]:
    problem = normalize_text(row["problem"])
    solution = build_solution(row, final_answer_prefix)
    example_index = row["example_index"]
    converted = {
        "id": f"math-{example_index}",
        "example_index": example_index,
        "source": "math",
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
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    val_count = max(1, math.floor(total * val_size))
    test_count = max(1, math.floor(total * test_size))
    train_rows = shuffled[val_count + test_count :]
    val_rows = shuffled[:val_count]
    test_rows = shuffled[val_count : val_count + test_count]
    return train_rows, val_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict[str, Any]]) -> None:
    sources = Counter(row["source"] for row in rows)
    print(f"{name}: {len(rows)} examples, sources={dict(sources)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../normalized_outputs/solver_full_trajectory_dataset.jsonl")
    parser.add_argument("--normalized-output", default="../normalized_outputs/solver_math_only_final_answer_dataset.jsonl")
    parser.add_argument(
        "--skip-normalized-output",
        action="store_true",
        help="Do not write a filtered normalized JSONL copy; only create SFT splits.",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-name", default="train_sft.jsonl")
    parser.add_argument("--val-name", default="val_sft.jsonl")
    parser.add_argument("--test-name", default="test_sft.jsonl")
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--final-answer-prefix", default="Final Answer:")
    args = parser.parse_args()

    raw_rows = read_jsonl(Path(args.input))
    math_rows = filter_math_rows(raw_rows, args.limit)
    if not math_rows:
        raise ValueError("No MATH rows with final_answer found")
    if args.limit is not None and len(math_rows) < args.limit:
        print(
            f"WARNING: requested --limit {args.limit}, but only found "
            f"{len(math_rows)} unique MATH rows with final_answer."
        )

    if args.skip_normalized_output:
        print("Skipped writing a filtered normalized JSONL copy.")
    else:
        normalized_output = Path(args.normalized_output)
        write_jsonl(normalized_output, math_rows)
        print(f"Wrote MATH-only normalized rows to {normalized_output.resolve()}")

    converted = [
        convert_row(
            row,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
            final_answer_prefix=args.final_answer_prefix,
        )
        for row in math_rows
    ]
    train_rows, val_rows, test_rows = split_rows(converted, args.val_size, args.test_size, args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / args.train_name, train_rows)
    write_jsonl(output_dir / args.val_name, val_rows)
    write_jsonl(output_dir / args.test_name, test_rows)

    summarize("normalized_math", math_rows)
    summarize("train", train_rows)
    summarize("val", val_rows)
    summarize("test", test_rows)
    print(f"Wrote SFT splits to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
