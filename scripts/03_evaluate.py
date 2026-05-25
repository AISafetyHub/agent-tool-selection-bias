#!/usr/bin/env python3
"""Step 3: Run evaluation across models."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.simulator import run_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "system_prompts"


def main():
    parser = argparse.ArgumentParser(description="Run multi-turn evaluation")
    parser.add_argument("--benchmark", default="data/validated/benchmark.jsonl")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs or 'all'")
    parser.add_argument("--system-prompt", default="default", choices=["default", "privilege_aware"])
    parser.add_argument("--output", default="data/eval_logs")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--error-strategy", default="first_attempt_fail")
    parser.add_argument("--limit", type=int, default=None, help="Limit scenarios per model (for testing)")
    args = parser.parse_args()

    # Load model list
    with open(CONFIGS_DIR / "models.yaml") as f:
        all_models = yaml.safe_load(f)["models"]

    if args.models == "all":
        models = [m["id"] for m in all_models if not m.get("optional")]
    else:
        models = [m.strip() for m in args.models.split(",")]

    system_prompt_path = PROMPTS_DIR / f"agent_{args.system_prompt}.txt"
    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output)

    for model_id in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*60}")
        results = run_batch(
            benchmark_path=benchmark_path,
            model=model_id,
            system_prompt_path=system_prompt_path,
            output_dir=output_dir,
            max_turns=args.max_turns,
            error_strategy=args.error_strategy,
            limit=args.limit,
        )
        if results:
            over_privileged = sum(1 for row in results if row.get("over_privileged_selection"))
            risk_called = sum(1 for row in results if row.get("first_risk_turn") is not None)
            print(f"Risk tool called: {risk_called}/{len(results)}")
            print(f"Over-privileged selection: {over_privileged}/{len(results)}")


if __name__ == "__main__":
    main()
