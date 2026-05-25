#!/usr/bin/env python3
"""Prepare repair candidates and prompts from validation outputs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.synthesis.repair import build_repair_candidates


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build repair prompts from validation artifacts")
    parser.add_argument("--scenarios", default="data/validated/scenarios_deduped.jsonl")
    parser.add_argument("--bias-report", default="data/validated/bias_report.json")
    parser.add_argument("--diversity-report", default="data/validated/diversity_report.json")
    parser.add_argument("--failure-analysis", default="data/validated/failure_analysis.json")
    parser.add_argument("--output", default="data/validated/repair_candidates.json")
    parser.add_argument("--prompt-dir", default="data/validated/repair_prompts")
    args = parser.parse_args()

    scenarios = _load_jsonl(Path(args.scenarios))
    bias_report = _load_json(Path(args.bias_report))
    diversity_report = _load_json(Path(args.diversity_report))

    failure_path = Path(args.failure_analysis)
    failure_analysis = _load_json(failure_path) if failure_path.exists() else []

    candidates = build_repair_candidates(
        scenarios=scenarios,
        bias_reports=bias_report,
        failure_analysis=failure_analysis,
        diversity_reports=diversity_report,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    prompt_dir = Path(args.prompt_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        prompt_path = prompt_dir / f"{candidate['scenario_id']}.txt"
        with open(prompt_path, "w") as f:
            f.write(candidate["repair_prompt"])

    print(f"Prepared {len(candidates)} repair candidates -> {output_path}")
    print(f"Prompt directory -> {prompt_dir}")


if __name__ == "__main__":
    main()
