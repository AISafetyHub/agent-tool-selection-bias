#!/usr/bin/env python3
"""Validate an existing scenario directory of JSON files."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.validation.schema_check import validate_batch
from src.validation.bias_check import check_bias, check_diversity
from src.validation.failure_analysis import analyze_failures
from src.validation.agreement import compute_agreement, filter_benchmark
from src.validation.sufficiency_judge_v2 import judge_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_scenarios_from_dir(input_dir: Path) -> list[dict]:
    scenarios = []
    for path in sorted(input_dir.glob("*.json")):
        with open(path) as f:
            scenarios.append(json.load(f))
    return scenarios


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate an existing scenario directory")
    parser.add_argument("--input-dir", default="data/raw_xhub_40", help="Directory containing scenario JSON files")
    parser.add_argument("--output-dir", default="data/validated_existing", help="Directory to save validation outputs")
    parser.add_argument("--skip-judge", action="store_true", help="Skip sufficiency judging and save only precheck outputs")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="Same-cell duplicate threshold")
    parser.add_argument(
        "--judge-models",
        default=None,
        help="Comma-separated judge models override, e.g. 'gpt-5.4,gpt-4o'",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        cfg = yaml.safe_load(f)

    scenarios = load_scenarios_from_dir(input_dir)
    if not scenarios:
        raise RuntimeError(f"No JSON scenarios found in {input_dir}")

    raw_jsonl_path = output_dir / "scenarios_input.jsonl"
    save_jsonl(raw_jsonl_path, scenarios)

    schema_result = validate_batch(scenarios)
    valid_scenarios = schema_result["valid"]
    print(f"Schema check: {len(valid_scenarios)} valid / {len(scenarios)} total")

    if schema_result["errors"]:
        err_path = output_dir / "schema_errors.json"
        with open(err_path, "w") as f:
            json.dump(schema_result["errors"], f, indent=2)
        print(f"Schema errors -> {err_path}")

    valid_jsonl_path = output_dir / "scenarios_schema_valid.jsonl"
    save_jsonl(valid_jsonl_path, valid_scenarios)

    bias_result = check_bias(valid_scenarios)
    bias_report_path = output_dir / "bias_report.json"
    with open(bias_report_path, "w") as f:
        json.dump(bias_result["reports"], f, indent=2)
    print(f"Bias pre-check: {len(bias_result['passed'])} passed / {len(valid_scenarios)} schema-valid")

    diversity_result = check_diversity(bias_result["passed"], similarity_threshold=args.similarity_threshold)
    diversity_report_path = output_dir / "diversity_report.json"
    with open(diversity_report_path, "w") as f:
        json.dump(diversity_result["reports"], f, indent=2)
    print(f"Diversity pre-check: {len(diversity_result['passed'])} passed / {len(bias_result['passed'])} bias-clean")

    prechecked = diversity_result["passed"]
    prechecked_jsonl_path = output_dir / "scenarios_prechecked.jsonl"
    save_jsonl(prechecked_jsonl_path, prechecked)

    if args.skip_judge:
        benchmark_path = output_dir / "benchmark.jsonl"
        save_jsonl(benchmark_path, prechecked)
        report = {
            "input_count": len(scenarios),
            "schema_valid_count": len(valid_scenarios),
            "bias_pass_count": len(bias_result["passed"]),
            "bias_flagged_count": len(bias_result["flagged"]),
            "diversity_pass_count": len(diversity_result["passed"]),
            "diversity_flagged_count": len(diversity_result["flagged"]),
        }
        report_path = output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved {len(prechecked)} prechecked scenarios -> {benchmark_path}")
        print(f"Report -> {report_path}")
        return

    if args.judge_models:
        judge_models = [item.strip() for item in args.judge_models.split(",") if item.strip()]
    else:
        judge_models = cfg["validation"]["judge_models"]

    judgments_path = output_dir / "judgments_v2.jsonl"
    judgments = judge_batch(
        prechecked,
        judge_models=judge_models,
        output_path=judgments_path,
        temperature=cfg["validation"].get("judge_temperature", 0.0),
    )

    failure_analysis = analyze_failures(prechecked, judgments, judge_models)
    failure_path = output_dir / "failure_analysis.json"
    with open(failure_path, "w") as f:
        json.dump(failure_analysis, f, indent=2)

    agreement = compute_agreement(judgments, judge_models)
    filtered, filter_stats = filter_benchmark(prechecked, judgments, judge_models)
    benchmark_path = output_dir / "benchmark.jsonl"
    save_jsonl(benchmark_path, filtered)

    report = {
        "input_count": len(scenarios),
        "schema_valid_count": len(valid_scenarios),
        "bias_pass_count": len(bias_result["passed"]),
        "bias_flagged_count": len(bias_result["flagged"]),
        "diversity_pass_count": len(diversity_result["passed"]),
        "diversity_flagged_count": len(diversity_result["flagged"]),
        **filter_stats,
        "agreement": agreement,
        "judge_models": judge_models,
    }
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Benchmark: {len(filtered)} scenarios -> {benchmark_path}")
    print(f"Agreement: kappa={agreement['kappa']}, rate={agreement['agreement_rate']:.1%}")
    print(f"Failure analysis -> {failure_path}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
