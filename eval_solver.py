#!/usr/bin/env python3
"""Evaluate the trained solver on processed SFT JSONL data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def extract_answer(text: str) -> Optional[str]:
    final_match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        return final_match.group(1).strip().splitlines()[0].strip()
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return None


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


def generate_solution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="outputs/smollm2-135m-solver-sft",
        help="Path to the trained full model, or a base model id when using --adapter.",
    )
    parser.add_argument(
        "--adapter",
        help="Optional PEFT adapter path. Leave unset for the default full fine-tuned SmolLM2 model.",
    )
    parser.add_argument("--test-file", default="test_sft.jsonl")
    parser.add_argument("--output-file", default="eval_predictions.jsonl")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.test_file))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    model, tokenizer = load_model_and_tokenizer(args)

    scored = 0
    correct = 0
    with Path(args.output_file).open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows, 1):
            prediction = generate_solution(
                model=model,
                tokenizer=tokenizer,
                messages=row["messages"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            expected_answer = (row.get("final_answer") or "").strip()
            predicted_answer = extract_answer(prediction)
            is_correct = None
            if expected_answer and predicted_answer:
                scored += 1
                is_correct = normalize_answer(expected_answer) == normalize_answer(predicted_answer)
                correct += int(is_correct)

            record = {
                "id": row.get("id"),
                "source": row.get("source"),
                "problem": row.get("problem"),
                "expected_solution": row.get("solution"),
                "expected_final_answer": expected_answer,
                "prediction": prediction,
                "predicted_final_answer": predicted_answer,
                "answer_exact_match": is_correct,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row.get('id')} exact_match={is_correct}")

    if scored:
        print(f"Final-answer exact match: {correct}/{scored} = {correct / scored:.3f}")
    else:
        print("No examples with both expected and predicted final answers were scored.")
    print(f"Wrote predictions to {args.output_file}")


if __name__ == "__main__":
    main()
