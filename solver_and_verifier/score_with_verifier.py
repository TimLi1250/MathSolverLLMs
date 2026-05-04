#!/usr/bin/env python3
"""Smoke-test the verifier reward on SFT JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reward_model import VerifierRewardModel


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("problem") and row.get("solution"):
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-dir", default="../verifier/modernbert_joint_verifier_best")
    parser.add_argument("--input-file", default="../newest_solver/data/test_sft.jsonl")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_file), limit=args.limit)
    if not rows:
        raise ValueError(f"No rows with problem and solution found in {args.input_file}")

    reward_model = VerifierRewardModel.from_pretrained(
        args.verifier_dir,
        max_length=args.max_length,
        device=args.device,
    )
    scores = reward_model.score_solutions(
        [row["problem"] for row in rows],
        [row["solution"] for row in rows],
        batch_size=args.batch_size,
    )

    for row, score in zip(rows, scores):
        problem_preview = " ".join(row["problem"].split())[:100]
        answer = row.get("final_answer", "")
        print(f"{row.get('id', '<no-id>')}\tscore={score:.4f}\tanswer={answer}\tproblem={problem_preview}")

    mean_score = sum(scores) / len(scores)
    print(f"\nMean verifier reward over {len(scores)} row(s): {mean_score:.4f}")


if __name__ == "__main__":
    main()

