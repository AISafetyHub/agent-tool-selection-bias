#!/usr/bin/env python3
"""Run evaluation for a single scenario JSON file and save the full trajectory."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api_client import get_default_client
from src.evaluation.simulator import simulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "system_prompts"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "eval_one_case"


def main():
    parser = argparse.ArgumentParser(description="Evaluate one scenario and save full trajectory")
    parser.add_argument("--case", required=True, help="Path to one scenario JSON file")
    parser.add_argument("--model", default="gpt-5.4", help="Model to evaluate")
    parser.add_argument("--system-prompt", default="default", choices=["default", "privilege_aware"])
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--error-strategy", default="first_attempt_fail")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    case_path = Path(args.case)
    scenario = json.loads(case_path.read_text())
    system_prompt_path = PROMPTS_DIR / f"agent_{args.system_prompt}.txt"
    system_prompt = system_prompt_path.read_text().strip()

    result = simulate(
        scenario=scenario,
        model=args.model,
        system_prompt=system_prompt,
        max_turns=args.max_turns,
        error_strategy=args.error_strategy,
        client=get_default_client(),
    )

    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{scenario['scenario_id']}__{args.model.replace('/', '__')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved trajectory -> {output_path}")
    print(f"Scenario: {result['scenario_id']}")
    print(f"Model: {result['model']}")
    print(f"Turns: {result['total_turns']}")
    print(f"Termination: {result['termination_reason']}")
    print(f"First risk turn: {result['first_risk_turn']}")
    print(f"Over-privileged selection: {result['over_privileged_selection']}")
    print(
        "Usage: "
        f"prompt={result['usage']['prompt_tokens']} "
        f"completion={result['usage']['completion_tokens']} "
        f"total={result['usage']['total_tokens']}"
    )


if __name__ == "__main__":
    main()
