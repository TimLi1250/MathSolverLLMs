#!/usr/bin/env python3
"""Run inference with a trained SmolLM2 solver model or optional PEFT adapter."""

from __future__ import annotations

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Provide a complete solution "
    "with all necessary reasoning."
)
USER_TEMPLATE = "Solve the following problem:\n\n{problem}"


def read_problem(args: argparse.Namespace) -> str:
    if args.problem:
        return args.problem.strip()
    if args.problem_file:
        with open(args.problem_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return input("Problem: ").strip()


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
    parser.add_argument("--problem")
    parser.add_argument("--problem-file")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    problem = read_problem(args)
    if not problem:
        raise SystemExit("No problem provided.")

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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
