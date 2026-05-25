#!/usr/bin/env python3
"""Run LoRA SFT with TRL for tool-calling chat datasets.

Supports records with:
  - messages: OpenAI-style chat messages
  - tools:    OpenAI-style function schemas (optional)

Key difference from train_sft_trl.py:
  - Passes tools= to apply_chat_template so tool definitions appear in the
    system prompt exactly as they do at eval time.
  - Defaults assistant_only_loss=True.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _render_one_example(tokenizer, example: dict[str, Any]) -> str:
    """Render one training example, passing tools= when available.

    Training data has intermediate assistant turns with content wrapped in
    <think>reasoning</think> followed by tool_calls, and a plain-text final
    turn with no tool_calls.  Passing tools= ensures the system prompt
    includes tool definitions, matching the eval-time format exactly.
    """
    messages = example.get("messages")
    if not messages:
        raise ValueError("Missing required field: messages")

    tools = example.get("tools")
    kwargs = dict(tokenize=False, add_generation_prompt=False)

    if tools:
        try:
            return tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, **kwargs)


def render_chat_examples(dataset, tokenizer):
    def convert(example):
        return {"text": _render_one_example(tokenizer, example)}

    return dataset.map(convert)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["training"]["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        device_map="auto",
    )

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = (
        load_dataset("json", data_files=args.eval_data, split="train")
        if args.eval_data
        else None
    )

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )

    train_dataset = render_chat_examples(train_dataset, tokenizer)
    if eval_dataset is not None:
        eval_dataset = render_chat_examples(eval_dataset, tokenizer)

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"].get("eval_steps"),
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        dataset_text_field="text",
        max_length=cfg["training"]["max_seq_length"],
        report_to=cfg["training"]["report_to"],
        assistant_only_loss=cfg["training"].get("assistant_only_loss", True),
        packing=cfg["training"].get("packing", False),
        eval_strategy=(
            "steps"
            if eval_dataset is not None and cfg["training"].get("eval_steps")
            else "no"
        ),
        save_strategy="steps",
        max_steps=args.max_steps if args.max_steps is not None else -1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{output_dir.rstrip('/')}/final_adapter")
    tokenizer.save_pretrained(f"{output_dir.rstrip('/')}/final_adapter")

    print(
        json.dumps(
            {
                "train_data": args.train_data,
                "eval_data": args.eval_data,
                "output_dir": output_dir,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
                "max_steps": args.max_steps,
                "assistant_only_loss": training_args.assistant_only_loss,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
