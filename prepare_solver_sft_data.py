#!/usr/bin/env python3
"""Prepare solver full-trajectory data for chat SFT.

Reads the raw solver JSONL, validates each row, lightly cleans source-specific
markup, concatenates solution steps, builds chat messages, and writes
train/validation/test JSONL files.
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
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            validate_raw_row(row, line_no)
            rows.append(row)
    return rows


def validate_raw_row(row: dict[str, Any], line_no: int) -> None:
    required = ["problem", "solution_steps", "source"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Line {line_no}: missing required keys: {missing}")
    if not isinstance(row["problem"], str) or not row["problem"].strip():
        raise ValueError(f"Line {line_no}: problem must be a non-empty string")
    if not isinstance(row["solution_steps"], list) or not row["solution_steps"]:
        raise ValueError(f"Line {line_no}: solution_steps must be a non-empty list")
    if not all(isinstance(step, str) and step.strip() for step in row["solution_steps"]):
        raise ValueError(f"Line {line_no}: every solution step must be a non-empty string")
    if "final_answer" in row and row["final_answer"] is not None and not isinstance(row["final_answer"], str):
        raise ValueError(f"Line {line_no}: final_answer must be a string when present")


def convert_literal_newlines(text: str) -> str:
    return re.sub(r"\\n(?=([A-Z0-9\[\]\{\}:$\\]|\s|$))", "\n", text)


def normalize_text(text: str, convert_escaped_newlines: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if convert_escaped_newlines:
        text = convert_literal_newlines(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def replace_wiki_link(match: re.Match[str]) -> str:
    body = match.group(1)
    if "|" in body:
        return body.rsplit("|", 1)[1]
    return body


def replace_eqn_template(match: re.Match[str]) -> str:
    body = match.group(1)
    fields: dict[str, str] = {}
    for part in re.split(r"\s*\|\s*", body):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()

    left = fields.get("l", "")
    right = fields.get("r", "")
    comment = fields.get("c", "")

    if left and right:
        rendered = f"{left} = {right}"
    else:
        rendered = left or right

    if comment:
        rendered = f"{rendered} ({comment})" if rendered else comment
    return rendered


def clean_naturalproofs(text: str) -> str:
    text = re.sub(r"\[\[([^\]]+)\]\]", replace_wiki_link, text)
    text = text.replace("'''", "")
    text = text.replace("{{begin-eqn}}", "\n")
    text = text.replace("{{end-eqn}}", "\n")
    text = re.sub(r"\{\{eqn\s*\|(.*?)\}\}", replace_eqn_template, text, flags=re.DOTALL)
    text = re.sub(r"\{\{math\|([^{}]*)\}\}", r"\1", text)
    text = re.sub(r"\{\{m\|([^{}]*)\}\}", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text


def clean_proofnet(text: str) -> str:
    text = re.sub(r"^\s*\\begin\{proof\}\s*", "", text)
    text = re.sub(r"\s*\\end\{proof\}\s*$", "", text)
    return text


def clean_text(text: str, source: str) -> str:
    text = normalize_text(text, convert_escaped_newlines=source == "naturalproofs")
    if source == "naturalproofs":
        text = clean_naturalproofs(text)
    elif source == "proofnet":
        text = clean_proofnet(text)
    return normalize_text(text)


def build_solution(row: dict[str, Any], append_final_answer: bool, final_answer_prefix: str) -> str:
    source = row.get("source", "")
    steps = [clean_text(step, source) for step in row["solution_steps"]]
    solution = "\n\n".join(step for step in steps if step)
    final_answer = (row.get("final_answer") or "").strip()
    if append_final_answer and final_answer:
        solution = f"{solution}\n\n{final_answer_prefix} {clean_text(final_answer, source)}"
    return solution.strip()


def row_id(row: dict[str, Any], fallback_index: int) -> str:
    source = row.get("source", "example")
    example_index = row.get("example_index", fallback_index)
    return f"{source}-{example_index}"


def convert_row(
    row: dict[str, Any],
    fallback_index: int,
    append_final_answer: bool,
    final_answer_prefix: str,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    source = row.get("source", "")
    problem = clean_text(row["problem"], source)
    solution = build_solution(row, append_final_answer, final_answer_prefix)
    user_content = user_template.format(problem=problem)

    converted = {
        "id": row_id(row, fallback_index),
        "example_index": row.get("example_index", fallback_index),
        "source": source,
        "problem": problem,
        "solution": solution,
        "final_answer": (row.get("final_answer") or "").strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
        buckets[row.get("source", "<missing>")].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for bucket_rows in buckets.values():
        shuffled = list(bucket_rows)
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_split_summary(name: str, rows: list[dict[str, Any]]) -> None:
    sources = Counter(row.get("source", "<missing>") for row in rows)
    answer_count = sum(1 for row in rows if row.get("final_answer"))
    print(f"{name}: {len(rows)} examples, final answers={answer_count}, sources={dict(sources)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="solver_full_trajectory_dataset.jsonl")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--train-name", default="train_sft.jsonl")
    parser.add_argument("--val-name", default="val_sft.jsonl")
    parser.add_argument("--test-name", default="test_sft.jsonl")
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    parser.add_argument("--final-answer-prefix", default="Final answer:")
    parser.add_argument(
        "--no-append-final-answer",
        action="store_true",
        help="Do not append non-empty final_answer values to the assistant trajectory.",
    )
    args = parser.parse_args()

    raw_rows = read_jsonl(Path(args.input))
    converted_rows = [
        convert_row(
            row=row,
            fallback_index=index,
            append_final_answer=not args.no_append_final_answer,
            final_answer_prefix=args.final_answer_prefix,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
        )
        for index, row in enumerate(raw_rows)
    ]

    train_rows, val_rows, test_rows = split_rows(
        converted_rows,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / args.train_name, train_rows)
    write_jsonl(output_dir / args.val_name, val_rows)
    write_jsonl(output_dir / args.test_name, test_rows)

    print_split_summary("train", train_rows)
    print_split_summary("val", val_rows)
    print_split_summary("test", test_rows)


if __name__ == "__main__":
    main()
