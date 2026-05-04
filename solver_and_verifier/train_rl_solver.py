#!/usr/bin/env python3
"""RL fine-tuning for the MATH-only solver using the verifier as reward.

This is intentionally lightweight: it uses a REINFORCE-style objective with an
EMA baseline and a KL-style penalty against the frozen MATH-only SFT adapter.
The reward is the verifier's P(label=1) for the generated full solution.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from reward_model import VerifierRewardModel


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Provide a complete solution "
    "with all necessary reasoning, and end with a line of the form "
    "'Final Answer: <answer>'."
)
DEFAULT_USER_TEMPLATE = "Solve the following problem:\n\n{problem}"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("problem"):
                raise ValueError(f"{path}:{line_no}: missing problem")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_prompt(row: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        return tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)

    problem = str(row["problem"]).strip()
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(problem=problem)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def parse_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_solver_with_adapter(
    base_model_name: str,
    adapter_path: str,
    dtype: torch.dtype,
    device_map: str | None,
    is_trainable: bool,
) -> torch.nn.Module:
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Missing solver adapter: {adapter_dir}. Train newest_solver first or pass --solver-adapter."
        )

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device_map and device_map != "none":
        kwargs["device_map"] = device_map

    base = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
    model.config.use_cache = True
    if not is_trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def make_full_attention_mask(
    sequences: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_width: int,
    pad_token_id: int,
) -> torch.Tensor:
    attention_mask = torch.ones_like(sequences, dtype=torch.long)
    attention_mask[:, :prompt_width] = prompt_attention_mask
    attention_mask[:, prompt_width:] = (sequences[:, prompt_width:] != pad_token_id).long()
    return attention_mask


def make_response_mask(sequences: torch.Tensor, prompt_width: int, pad_token_id: int) -> torch.Tensor:
    response_mask = torch.zeros_like(sequences, dtype=torch.bool)
    response_mask[:, prompt_width:] = sequences[:, prompt_width:] != pad_token_id
    return response_mask


def sequence_logprobs(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = sequences[:, 1:]
    token_mask = response_mask[:, 1:].float()

    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    token_counts = token_mask.sum(dim=-1).clamp(min=1.0)
    summed = (token_logprobs * token_mask).sum(dim=-1)
    meaned = summed / token_counts
    return summed, meaned, token_counts, token_logprobs


def split_solution_steps(solution: str, max_steps: int) -> list[str]:
    text = solution.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    steps = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(steps) <= 1:
        steps = [line.strip() for line in text.splitlines() if line.strip()]
    return steps[:max_steps]


def score_next_step_shaping(
    reward_model: VerifierRewardModel,
    problems: list[str],
    responses: list[str],
    batch_size: int,
    max_steps: int,
) -> list[float]:
    all_problems: list[str] = []
    all_prefixes: list[list[str]] = []
    all_targets: list[str] = []
    owners: list[int] = []

    for owner, (problem, response) in enumerate(zip(problems, responses)):
        prefix: list[str] = []
        for step in split_solution_steps(response, max_steps=max_steps):
            all_problems.append(problem)
            all_prefixes.append(list(prefix))
            all_targets.append(step)
            owners.append(owner)
            prefix.append(step)

    if not all_targets:
        return [0.0 for _ in responses]

    step_scores = reward_model.score_next_steps(
        all_problems,
        all_prefixes,
        all_targets,
        batch_size=batch_size,
    )

    totals = [0.0 for _ in responses]
    counts = [0 for _ in responses]
    for owner, score in zip(owners, step_scores):
        totals[owner] += score
        counts[owner] += 1
    return [totals[i] / counts[i] if counts[i] else 0.0 for i in range(len(responses))]


def save_adapter(model: torch.nn.Module, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--solver-adapter", default="../newest_solver/outputs/qwen2.5-0.5b-math-only-lora")
    parser.add_argument("--verifier-dir", default="../verifier/modernbert_joint_verifier_best")
    parser.add_argument("--train-file", default="../newest_solver/data/train_sft.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-0.5b-math-verifier-rl-lora")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--baseline-ema-decay", type=float, default=0.90)
    parser.add_argument("--initial-baseline", type=float, default=0.50)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--reward-max-length", type=int, default=None)
    parser.add_argument("--next-step-reward-weight", type=float, default=0.0)
    parser.add_argument("--next-step-max-steps", type=int, default=12)

    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--reference-device-map", default="auto")
    parser.add_argument("--reward-device", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    if not 0.0 <= args.next_step_reward_weight <= 1.0:
        raise ValueError("--next-step-reward-weight must be between 0 and 1")

    set_seed(args.seed)
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"

    rows = read_jsonl(Path(args.train_file), limit=args.max_train_examples)
    if not rows:
        raise ValueError(f"No training rows loaded from {args.train_file}")

    dtype = parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    print("Loading policy solver adapter...")
    policy = load_solver_with_adapter(
        args.base_model,
        args.solver_adapter,
        dtype=dtype,
        device_map=args.device_map,
        is_trainable=True,
    )
    policy.eval()
    policy.print_trainable_parameters()

    print("Loading frozen reference solver adapter...")
    reference = load_solver_with_adapter(
        args.base_model,
        args.solver_adapter,
        dtype=dtype,
        device_map=args.reference_device_map,
        is_trainable=False,
    )

    print("Loading verifier reward model...")
    reward_model = VerifierRewardModel.from_pretrained(
        args.verifier_dir,
        max_length=args.reward_max_length,
        device=args.reward_device,
    )

    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Policy model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    baseline = float(args.initial_baseline)
    optimizer.zero_grad(set_to_none=True)

    for update_idx in range(1, args.num_updates + 1):
        started_at = time.time()
        batch_rows = [rows[rng.randrange(len(rows))] for _ in range(args.train_batch_size)]
        problems = [str(row["problem"]).strip() for row in batch_rows]
        prompts = [build_prompt(row, tokenizer) for row in batch_rows]

        policy_device = model_input_device(policy)
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        encoded.pop("token_type_ids", None)
        encoded = {k: v.to(policy_device) for k, v in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]

        with torch.no_grad():
            generated = policy.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_token_ids = generated[:, prompt_width:]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)

        full_attention_mask = make_full_attention_mask(
            generated,
            encoded["attention_mask"],
            prompt_width=prompt_width,
            pad_token_id=tokenizer.pad_token_id,
        )
        response_mask = make_response_mask(
            generated,
            prompt_width=prompt_width,
            pad_token_id=tokenizer.pad_token_id,
        )

        solution_scores = reward_model.score_solutions(
            problems,
            responses,
            batch_size=args.reward_batch_size,
        )

        if args.next_step_reward_weight > 0.0:
            step_scores = score_next_step_shaping(
                reward_model=reward_model,
                problems=problems,
                responses=responses,
                batch_size=args.reward_batch_size,
                max_steps=args.next_step_max_steps,
            )
            rewards_list = [
                (1.0 - args.next_step_reward_weight) * solution_score
                + args.next_step_reward_weight * step_score
                for solution_score, step_score in zip(solution_scores, step_scores)
            ]
        else:
            step_scores = None
            rewards_list = solution_scores

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=policy_device)
        advantages = rewards - baseline
        baseline = (
            args.baseline_ema_decay * baseline
            + (1.0 - args.baseline_ema_decay) * float(rewards.detach().mean().cpu())
        )

        _, policy_logprob_mean, token_counts, policy_token_logprobs = sequence_logprobs(
            policy,
            generated,
            full_attention_mask,
            response_mask,
        )

        ref_device = model_input_device(reference)
        ref_generated = generated.to(ref_device)
        ref_attention_mask = full_attention_mask.to(ref_device)
        ref_response_mask = response_mask.to(ref_device)
        with torch.no_grad():
            _, _, _, ref_token_logprobs = sequence_logprobs(
                reference,
                ref_generated,
                ref_attention_mask,
                ref_response_mask,
            )
        ref_token_logprobs = ref_token_logprobs.to(policy_device)

        shifted_response_mask = response_mask[:, 1:].float()
        logratio = (policy_token_logprobs - ref_token_logprobs) * shifted_response_mask
        kl_surrogate = (logratio.pow(2).sum(dim=-1) / token_counts).mean()

        pg_loss = -(advantages.detach() * policy_logprob_mean).mean()
        loss = pg_loss + args.kl_coef * kl_surrogate
        (loss / args.gradient_accumulation_steps).backward()

        should_step = update_idx % args.gradient_accumulation_steps == 0
        grad_norm = None
        if should_step:
            grad_norm = float(clip_grad_norm_(trainable_params, args.max_grad_norm).detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = {
            "update": update_idx,
            "loss": float(loss.detach().cpu()),
            "pg_loss": float(pg_loss.detach().cpu()),
            "kl_surrogate": float(kl_surrogate.detach().cpu()),
            "reward_mean": float(rewards.detach().mean().cpu()),
            "reward_min": float(rewards.detach().min().cpu()),
            "reward_max": float(rewards.detach().max().cpu()),
            "solution_reward_mean": float(sum(solution_scores) / len(solution_scores)),
            "baseline": baseline,
            "mean_response_tokens": float(token_counts.detach().mean().cpu()),
            "grad_norm": grad_norm,
            "seconds": round(time.time() - started_at, 2),
        }
        if step_scores is not None:
            metrics["next_step_reward_mean"] = float(sum(step_scores) / len(step_scores))

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if update_idx % args.log_every == 0:
            print(json.dumps(metrics))
            sample_preview = " ".join(responses[0].split())[:240]
            print(f"sample_response: {sample_preview}")

        if args.save_every > 0 and update_idx % args.save_every == 0:
            save_adapter(policy, tokenizer, output_dir / f"checkpoint-{update_idx}")

    save_adapter(policy, tokenizer, output_dir)
    print(f"Saved RL-tuned solver adapter to {output_dir}")


if __name__ == "__main__":
    main()
