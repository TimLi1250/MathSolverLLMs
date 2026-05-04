#!/usr/bin/env python3
"""Verifier reward-model wrapper.

The formatting mirrors verifier/modernbert_joint_verifier_colab_prefix_truncation.ipynb.
The reward is P(label=1), where label 1 means correct_or_valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MAX_LENGTH = 2048
PREFIX_TRUNCATION_SAFETY_TOKENS = 32

SOLUTION_INSTRUCTION = "Verify whether the following full solution is correct."
NEXT_STEP_INSTRUCTION = (
    "Verify whether the target step is mathematically valid given the problem and the previous steps."
)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_prefix_steps(prefix_steps: Any) -> list[str]:
    if prefix_steps is None:
        return []
    if isinstance(prefix_steps, list):
        return [normalize_text(s) for s in prefix_steps if normalize_text(s)]
    text = normalize_text(prefix_steps)
    return [text] if text else []


def load_saved_max_length(model_dir: str | Path, fallback: int = DEFAULT_MAX_LENGTH) -> int:
    metadata_path = Path(model_dir) / "verifier_training_metadata.json"
    if not metadata_path.exists():
        return fallback
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return int(metadata.get("max_length", fallback))


def count_tokens(text: str, tokenizer, add_special_tokens: bool = False) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])


def format_solution_example(example: dict[str, Any]) -> str:
    instruction = example["instruction"].strip()
    problem = example["problem"].strip()
    solution = example["solution"].strip()

    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Task: full-solution verification\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Solution:\n"
        f"{solution}\n\n"
        "Binary label meaning: 1 = correct solution, 0 = incorrect solution."
    )


def format_next_step_without_prefix_truncation(example: dict[str, Any]) -> str:
    instruction = example["instruction"].strip()
    problem = example["problem"].strip()
    target_step = example["target_step"].strip()
    prefix_steps = normalize_prefix_steps(example.get("prefix_steps", []))

    if prefix_steps:
        prefix_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(prefix_steps))
    else:
        prefix_text = "(No previous steps.)"

    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Task: next-step verification\n\n"
        "Target step:\n"
        f"{target_step}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Previous steps:\n"
        f"{prefix_text}\n\n"
        "Binary label meaning: 1 = valid/correct step, 0 = invalid/incorrect step."
    )


def format_next_step_with_recent_prefix(
    example: dict[str, Any],
    tokenizer,
    max_length: int,
    safety_tokens: int = PREFIX_TRUNCATION_SAFETY_TOKENS,
) -> tuple[str, dict[str, int]]:
    instruction = example["instruction"].strip()
    problem = example["problem"].strip()
    target_step = example["target_step"].strip()
    prefix_steps = normalize_prefix_steps(example.get("prefix_steps", []))

    header = (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Task: next-step verification\n\n"
        "Target step:\n"
        f"{target_step}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Previous steps, with the most recent available context preserved:\n"
    )
    footer = "\nBinary label meaning: 1 = valid/correct step, 0 = invalid/incorrect step."

    if not prefix_steps:
        return (
            header + "(No previous steps.)\n" + footer,
            {
                "num_prefix_steps_original": 0,
                "num_prefix_steps_kept": 0,
                "num_prefix_steps_dropped": 0,
                "core_too_long": 0,
            },
        )

    core_len = count_tokens(header + footer, tokenizer, add_special_tokens=True)
    available_for_prefix = max_length - core_len - safety_tokens

    if available_for_prefix <= 0:
        return (
            header
            + "[All previous steps omitted because the problem and target step already fill the context window.]\n"
            + footer,
            {
                "num_prefix_steps_original": len(prefix_steps),
                "num_prefix_steps_kept": 0,
                "num_prefix_steps_dropped": len(prefix_steps),
                "core_too_long": 1,
            },
        )

    kept_reversed: list[str] = []
    used = 0
    for original_idx in range(len(prefix_steps) - 1, -1, -1):
        step_text = f"{original_idx + 1}. {prefix_steps[original_idx]}\n"
        step_len = count_tokens(step_text, tokenizer, add_special_tokens=False)
        if used + step_len > available_for_prefix:
            break
        kept_reversed.append(step_text)
        used += step_len

    kept_steps = list(reversed(kept_reversed))
    kept_count = len(kept_steps)
    dropped_count = len(prefix_steps) - kept_count

    if kept_steps:
        prefix_text = "".join(kept_steps)
    else:
        prefix_text = "[All previous steps omitted because they do not fit in the context window.]\n"

    if dropped_count > 0:
        prefix_text = f"[Omitted {dropped_count} earliest previous step(s) due to context length.]\n" + prefix_text

    return (
        header + prefix_text + footer,
        {
            "num_prefix_steps_original": len(prefix_steps),
            "num_prefix_steps_kept": kept_count,
            "num_prefix_steps_dropped": dropped_count,
            "core_too_long": 0,
        },
    )


def format_example(
    example: dict[str, Any],
    tokenizer=None,
    max_length: int = DEFAULT_MAX_LENGTH,
    smart_truncate_next_step_prefix: bool = True,
) -> tuple[str, dict[str, int]]:
    task_type = example["task_type"]

    if task_type == "solution":
        return (
            format_solution_example(example),
            {
                "num_prefix_steps_original": 0,
                "num_prefix_steps_kept": 0,
                "num_prefix_steps_dropped": 0,
                "core_too_long": 0,
            },
        )

    if task_type == "next_step":
        if smart_truncate_next_step_prefix:
            if tokenizer is None:
                raise ValueError("tokenizer is required for smart next-step prefix truncation")
            return format_next_step_with_recent_prefix(example, tokenizer, max_length=max_length)
        return (
            format_next_step_without_prefix_truncation(example),
            {
                "num_prefix_steps_original": len(normalize_prefix_steps(example.get("prefix_steps", []))),
                "num_prefix_steps_kept": len(normalize_prefix_steps(example.get("prefix_steps", []))),
                "num_prefix_steps_dropped": 0,
                "core_too_long": 0,
            },
        )

    raise ValueError(f"Unknown task_type: {task_type}")


def format_solution_for_reward(problem: str, solution: str) -> str:
    text, _ = format_example(
        {
            "task_type": "solution",
            "instruction": SOLUTION_INSTRUCTION,
            "problem": problem,
            "prefix_steps": [],
            "target_step": "",
            "solution": solution,
        }
    )
    return text


def format_next_step_for_reward(
    problem: str,
    prefix_steps: Sequence[str],
    target_step: str,
    tokenizer,
    max_length: int,
) -> str:
    text, _ = format_example(
        {
            "task_type": "next_step",
            "instruction": NEXT_STEP_INSTRUCTION,
            "problem": problem,
            "prefix_steps": list(prefix_steps),
            "target_step": target_step,
            "solution": "",
        },
        tokenizer=tokenizer,
        max_length=max_length,
        smart_truncate_next_step_prefix=True,
    )
    return text


@dataclass
class VerifierRewardModel:
    tokenizer: Any
    model: torch.nn.Module
    max_length: int
    correct_label_id: int

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        max_length: int | None = None,
        device: str | None = None,
        batch_dtype: torch.dtype | None = None,
    ) -> "VerifierRewardModel":
        model_dir = Path(model_dir)
        if max_length is None:
            max_length = load_saved_max_length(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        if batch_dtype is None:
            batch_dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                torch_dtype=batch_dtype,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=batch_dtype)

        model.to(torch_device)
        model.eval()

        label2id = getattr(model.config, "label2id", {}) or {}
        correct_label_id = int(label2id.get("correct_or_valid", 1))
        return cls(
            tokenizer=tokenizer,
            model=model,
            max_length=int(max_length),
            correct_label_id=correct_label_id,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def score_texts(self, texts: Sequence[str], batch_size: int = 4) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, self.correct_label_id]

            scores.extend(probs.detach().cpu().tolist())
        return scores

    def score_solutions(
        self,
        problems: Sequence[str],
        solutions: Sequence[str],
        batch_size: int = 4,
    ) -> list[float]:
        if len(problems) != len(solutions):
            raise ValueError("problems and solutions must have the same length")
        texts = [format_solution_for_reward(problem, solution) for problem, solution in zip(problems, solutions)]
        return self.score_texts(texts, batch_size=batch_size)

    def score_next_steps(
        self,
        problems: Sequence[str],
        prefixes: Sequence[Sequence[str]],
        target_steps: Sequence[str],
        batch_size: int = 4,
    ) -> list[float]:
        if not (len(problems) == len(prefixes) == len(target_steps)):
            raise ValueError("problems, prefixes, and target_steps must have the same length")
        texts = [
            format_next_step_for_reward(
                problem=problem,
                prefix_steps=prefix,
                target_step=target_step,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            for problem, prefix, target_step in zip(problems, prefixes, target_steps)
        ]
        return self.score_texts(texts, batch_size=batch_size)

