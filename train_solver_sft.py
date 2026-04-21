#!/usr/bin/env python3
"""Train SmolLM2-135M-Instruct on processed solver trajectory JSONL files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    example_id: Optional[str]


class SolverSFTDataset(Dataset):
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
        }


class DataCollatorForSolverSFT:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [feature["attention_mask"] for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no}: missing messages field")
            rows.append(row)
    return rows


def encode_row(row: dict[str, Any], tokenizer: AutoTokenizer, max_seq_length: int) -> EncodedExample:
    messages = row["messages"]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    input_ids = full["input_ids"]
    labels = list(input_ids)
    labels[: min(prompt_len, len(labels))] = [IGNORE_INDEX] * min(prompt_len, len(labels))

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            f"{row.get('id')}: no assistant tokens left after truncation. "
            "Increase --max-seq-length or filter the example."
        )

    return EncodedExample(
        input_ids=input_ids,
        attention_mask=full["attention_mask"],
        labels=labels,
        example_id=row.get("id"),
    )


def load_encoded_dataset(path: Path, tokenizer: AutoTokenizer, max_seq_length: int) -> SolverSFTDataset:
    rows = read_jsonl(path)
    examples = [encode_row(row, tokenizer, max_seq_length) for row in rows]
    print(f"{path}: loaded {len(examples)} encoded examples")
    return SolverSFTDataset(examples)


def prepare_model_for_kbit_training_low_memory(
    model: AutoModelForCausalLM,
    use_gradient_checkpointing: bool,
) -> AutoModelForCausalLM:
    """Prepare a 4-bit model for LoRA without PEFT's fp32 casting pass.

    `prepare_model_for_kbit_training` is the usual helper, but its fp32 cast can
    OOM on small GPUs. For optional quantized LoRA training we still need the
    important parts: freeze base weights and make input embeddings require
    gradients when gradient checkpointing is enabled.
    """
    for param in model.parameters():
        param.requires_grad = False

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    return model


def build_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    quantization_config = None
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto" if args.qlora else None,
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False
        if not args.qlora:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

    if args.qlora:
        if args.kbit_prepare_mode == "peft":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        elif args.kbit_prepare_mode == "low_memory":
            model = prepare_model_for_kbit_training_low_memory(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Unsupported --kbit-prepare-mode: {args.kbit_prepare_mode}")

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--train-file", default="train_sft.jsonl")
    parser.add_argument("--val-file", default="val_sft.jsonl")
    parser.add_argument("--output-dir", default="outputs/smollm2-135m-solver-sft")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument(
        "--kbit-prepare-mode",
        choices=["low_memory", "peft"],
        default="low_memory",
        help=(
            "Use low_memory on small GPUs to avoid PEFT's fp32 casting OOM. "
            "Use peft for the standard prepare_model_for_kbit_training path."
        ),
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_encoded_dataset(Path(args.train_file), tokenizer, args.max_seq_length)
    val_dataset = load_encoded_dataset(Path(args.val_file), tokenizer, args.max_seq_length)
    model = build_model(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSolverSFT(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
