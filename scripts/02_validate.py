#!/usr/bin/env python3
"""Step 2: Validate scenarios — schema check + sufficiency judging + filtering."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.synthesis.dedup import filter_and_dedup
from src.validation.schema_check import validate_batch
from src.validation.bias_check import check_bias, check_diversity
from src.validation.sufficiency_judge import judge_batch
from src.validation.failure_analysis import analyze_failures
from src.validation.agreement import compute_agreement, filter_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def main():
    parser = argparse.ArgumentParser(description="Validate and filter scenarios")
    parser.add_argument("--input", default="data/raw/scenarios_raw.jsonl", help="Raw scenarios")
    parser.add_argument("--output", default="data/validated", help="Output directory")
    parser.add_argument("--skip-judge", action="store_true", help="Skip sufficiency judging (schema only)")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="Same-cell duplicate threshold")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    # 1. Dedup
    dedup_path = output_dir / "scenarios_deduped.jsonl"
    dedup_stats = filter_and_dedup(input_path, dedup_path)

    # 2. Schema check
    scenarios = []
    with open(dedup_path) as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    result = validate_batch(scenarios)
    valid_scenarios = result["valid"]
    print(f"Schema check: {len(valid_scenarios)} valid / {len(scenarios)} total")

    if result["errors"]:
        err_path = output_dir / "schema_errors.json"
        with open(err_path, "w") as f:
            json.dump(result["errors"], f, indent=2)
        print(f"Schema errors saved to {err_path}")

    # 2.5 Bias and diversity pre-check
    bias_result = check_bias(valid_scenarios)
    prechecked_scenarios = bias_result["passed"]
    bias_report_path = output_dir / "bias_report.json"
    with open(bias_report_path, "w") as f:
        json.dump(bias_result["reports"], f, indent=2)
    print(f"Bias pre-check: {len(prechecked_scenarios)} passed / {len(valid_scenarios)} schema-valid")

    diversity_result = check_diversity(prechecked_scenarios, similarity_threshold=args.similarity_threshold)
    prechecked_scenarios = diversity_result["passed"]
    diversity_report_path = output_dir / "diversity_report.json"
    with open(diversity_report_path, "w") as f:
        json.dump(diversity_result["reports"], f, indent=2)
    print(f"Diversity pre-check: {len(prechecked_scenarios)} passed / {len(bias_result['passed'])} bias-clean")

    if args.skip_judge:
        # Save directly without judging
        benchmark_path = output_dir / "benchmark.jsonl"
        with open(benchmark_path, "w") as f:
            for s in prechecked_scenarios:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(prechecked_scenarios)} scenarios (no judging) -> {benchmark_path}")
        return

    # 3. Sufficiency judging
    judge_models = cfg["validation"]["judge_models"]
    judgments_path = output_dir / "judgments.jsonl"
    judgments = judge_batch(prechecked_scenarios, judge_models=judge_models, output_path=judgments_path)
    failure_analysis = analyze_failures(prechecked_scenarios, judgments, judge_models)
    failure_path = output_dir / "failure_analysis.json"
    with open(failure_path, "w") as f:
        json.dump(failure_analysis, f, indent=2)

    # 4. Agreement + filtering
    agreement = compute_agreement(judgments, judge_models)
    filtered, filter_stats = filter_benchmark(prechecked_scenarios, judgments, judge_models)

    # Save benchmark
    benchmark_path = output_dir / "benchmark.jsonl"
    with open(benchmark_path, "w") as f:
        for s in filtered:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save report
    report = {
        **dedup_stats,
        "schema_valid_count": len(valid_scenarios),
        "bias_pass_count": len(bias_result["passed"]),
        "bias_flagged_count": len(bias_result["flagged"]),
        "diversity_pass_count": len(diversity_result["passed"]),
        "diversity_flagged_count": len(diversity_result["flagged"]),
        **filter_stats,
        "agreement": agreement,
    }
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Benchmark: {len(filtered)} scenarios -> {benchmark_path}")
    print(f"Agreement: kappa={agreement['kappa']}, rate={agreement['agreement_rate']:.1%}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
