"""Enhanced two-judge sufficiency verification with scenario context and reporting."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api_client import chat_completion, extract_usage, get_xhub_client

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
PROMPT_TEMPLATE = (ROOT / "prompts" / "sufficiency_judge_v2.txt").read_text()
CONFIGS_DIR = ROOT / "configs"
LABELS = ["FULLY", "PARTIAL", "NO"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}


def _load_judge_config() -> dict:
    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        return yaml.safe_load(f)["validation"]


def _normalize_label(text: str, default: str | None = None) -> str | None:
    text = (text or "").strip().upper()
    if not text:
        logger.warning("Empty judge response")
        return default
    for label in LABELS:
        if label in text:
            return label
    logger.warning("Unexpected judge response: %r", text)
    return default


def judge_tool(
    user_instruction: str,
    panic_logic: str,
    tool: dict,
    model: str,
    client=None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    per_call_retries: int = 8,
    include_meta: bool = False,
) -> str | tuple[str, dict]:
    """Query one judge model for a single (task, tool) pair."""
    client = client or get_xhub_client()
    prompt = PROMPT_TEMPLATE.format(
        user_instruction=user_instruction,
        panic_logic=panic_logic or "",
        tool_name=tool["name"],
        tool_description=tool.get("description", ""),
        tool_parameters=json.dumps(tool.get("parameters", {}), ensure_ascii=False),
    )

    last_err = None
    for attempt in range(per_call_retries):
        try:
            response = chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                client=client,
            )
            label = _normalize_label(response.choices[0].message.content)
            if label is None:
                raise RuntimeError(f"empty_or_invalid_judge_response:{model}:{tool.get('name')}")
            if include_meta:
                return label, {
                    "model": model,
                    "tool_name": tool.get("name"),
                    "attempt": attempt + 1,
                    "usage": extract_usage(response),
                }
            return label
        except Exception as e:
            last_err = e
            wait = min(60, 5 * (2 ** attempt))
            logger.warning(
                "judge_tool retry %s/%s failed for scenario-tool=%s model=%s: %s; sleeping %ss",
                attempt + 1,
                per_call_retries,
                tool.get("name"),
                model,
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"judge_tool failed after {per_call_retries} retries for {tool.get('name')} / {model}: {last_err}")


def judge_scenario(
    scenario: dict,
    judge_models: list[str],
    client=None,
    temperature: float = 0.0,
    include_meta: bool = False,
) -> dict | tuple[dict, dict]:
    """Judge all tools in one scenario with all judge models.

    Returns {tool_name: {model_id: label}}.
    """
    client = client or get_xhub_client()
    results: dict[str, dict[str, str]] = {}
    usage_records = []
    for tool in scenario["tools"]:
        tool_result: dict[str, str] = {}
        for model in judge_models:
            judged = judge_tool(
                user_instruction=scenario["user_instruction"],
                panic_logic=scenario.get("panic_logic", ""),
                tool=tool,
                model=model,
                client=client,
                temperature=temperature,
                include_meta=include_meta,
            )
            if include_meta:
                label, meta = judged
                tool_result[model] = label
                usage_records.append(meta)
            else:
                tool_result[model] = judged
        results[tool["name"]] = tool_result
    if include_meta:
        summary = {
            "records": usage_records,
            "usage": {
                "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in usage_records),
                "completion_tokens": sum(r["usage"]["completion_tokens"] for r in usage_records),
                "total_tokens": sum(r["usage"]["total_tokens"] for r in usage_records),
            },
        }
        return results, summary
    return results


def judge_batch(
    scenarios: list[dict],
    judge_models: list[str],
    output_path: Path | None = None,
    resume: bool = True,
    temperature: float = 0.0,
) -> list[dict]:
    """Judge all scenarios and optionally append jsonl records for resume support."""
    client = get_xhub_client()

    existing: dict[str, dict] = {}
    if resume and output_path and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing[rec["scenario_id"]] = rec

    results = []
    fh = open(output_path, "a") if output_path else None
    try:
        for idx, scenario in enumerate(scenarios, start=1):
            sid = scenario["scenario_id"]
            if sid in existing:
                results.append(existing[sid])
                continue
            logger.info("[%s/%s] Judging %s", idx, len(scenarios), sid)
            judgments = judge_scenario(scenario, judge_models, client=client, temperature=temperature)
            rec = {
                "scenario_id": sid,
                "domain": scenario.get("domain"),
                "type": scenario.get("type"),
                "user_instruction": scenario.get("user_instruction"),
                "panic_logic": scenario.get("panic_logic"),
                "tools": scenario.get("tools"),
                "judgments": judgments,
            }
            results.append(rec)
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    return results


def load_scenarios(input_dir: Path) -> list[dict]:
    scenarios = []
    for path in sorted(input_dir.glob("*.json")):
        with open(path) as f:
            scenarios.append(json.load(f))
    return scenarios


def compute_agreement(judgments: list[dict], judge_models: list[str]) -> dict:
    if len(judge_models) != 2:
        raise ValueError("Exactly 2 judge models required")
    m1, m2 = judge_models
    labels_1: list[str] = []
    labels_2: list[str] = []
    pair_breakdown = Counter()

    for rec in judgments:
        for tool_name, model_labels in rec["judgments"].items():
            if m1 in model_labels and m2 in model_labels:
                a = model_labels[m1]
                b = model_labels[m2]
                labels_1.append(a)
                labels_2.append(b)
                pair_breakdown[f"{a}|{b}"] += 1

    total = len(labels_1)
    agree = sum(a == b for a, b in zip(labels_1, labels_2))
    p0 = agree / total if total else 0.0

    c1 = Counter(labels_1)
    c2 = Counter(labels_2)
    pe = sum((c1[label] / total) * (c2[label] / total) for label in LABELS) if total else 0.0
    kappa = (p0 - pe) / (1 - pe) if total and pe != 1.0 else 0.0

    return {
        "judge_models": judge_models,
        "total_pairs": total,
        "agreement_count": agree,
        "agreement_rate": round(p0, 4),
        "kappa": round(kappa, 4),
        "marginals": {m1: dict(c1), m2: dict(c2)},
        "pair_breakdown": dict(sorted(pair_breakdown.items())),
    }


def summarize_by_domain_type(scenarios: list[dict], judgments: list[dict], judge_models: list[str]) -> dict:
    judgment_map = {rec["scenario_id"]: rec["judgments"] for rec in judgments}
    out: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        key = f"{scenario['domain']} | {scenario['type']}"
        bucket = out.setdefault(key, {
            "domain": scenario["domain"],
            "escalation_type": scenario["type"],
            "scenario_count": 0,
            "tool_judgment_count": 0,
            "counts": {label: 0 for label in LABELS},
            "standard_case_insufficient_count": 0,
        })
        bucket["scenario_count"] += 1

        scenario_j = judgment_map[scenario["scenario_id"]]
        standard_case_bad = False
        for tool in scenario["tools"]:
            tool_labels = scenario_j[tool["name"]]
            for model in judge_models:
                label = tool_labels[model]
                bucket["counts"][label] += 1
                bucket["tool_judgment_count"] += 1
                if tool.get("role") == "standard" and label in {"PARTIAL", "NO"}:
                    standard_case_bad = True
        if standard_case_bad:
            bucket["standard_case_insufficient_count"] += 1

    for bucket in out.values():
        total = bucket["tool_judgment_count"] or 1
        bucket["ratios"] = {label: round(bucket["counts"][label] / total, 4) for label in LABELS}
        bucket["standard_case_insufficient_rate"] = round(
            bucket["standard_case_insufficient_count"] / bucket["scenario_count"], 4
        ) if bucket["scenario_count"] else 0.0
    return dict(sorted(out.items()))


def collect_problem_cases(scenarios: list[dict], judgments: list[dict], judge_models: list[str]) -> list[dict]:
    judgment_map = {rec["scenario_id"]: rec for rec in judgments}
    problem_cases = []

    for scenario in scenarios:
        rec = judgment_map[scenario["scenario_id"]]
        flagged_tools = []
        standard_insufficient = False
        for tool in scenario["tools"]:
            labels = rec["judgments"][tool["name"]]
            bad = {m: labels[m] for m in judge_models if labels[m] in {"PARTIAL", "NO"}}
            if bad:
                inferred_reason = "prompt_or_context_sensitive"
                if any(v == "NO" for v in bad.values()):
                    inferred_reason = "likely_capability_insufficient_or_irrelevant"
                elif tool.get("role") == "standard":
                    inferred_reason = "likely_capability_boundary_or_context-sensitive"
                flagged_tools.append({
                    "tool_name": tool["name"],
                    "role": tool.get("role"),
                    "all_labels": labels,
                    "flagged_labels": bad,
                    "inferred_reason": inferred_reason,
                    "description": tool.get("description", ""),
                })
                if tool.get("role") == "standard":
                    standard_insufficient = True
        if flagged_tools:
            problem_cases.append({
                "scenario_id": scenario["scenario_id"],
                "domain": scenario.get("domain"),
                "escalation_type": scenario.get("type"),
                "title": scenario.get("title"),
                "user_instruction": scenario.get("user_instruction"),
                "panic_logic": scenario.get("panic_logic"),
                "standard_case_insufficient": standard_insufficient,
                "flagged_tools": flagged_tools,
            })
    return problem_cases


def select_typical_failure_cases(scenarios: list[dict], judgments: list[dict], judge_models: list[str], max_cases: int = 5) -> list[dict]:
    judgment_map = {rec["scenario_id"]: rec["judgments"] for rec in judgments}
    candidates = []

    for scenario in scenarios:
        disagreement = 0
        bad_standard = 0
        bad_total = 0
        full_matrix = []
        for tool in scenario["tools"]:
            labels = judgment_map[scenario["scenario_id"]][tool["name"]]
            vals = [labels[m] for m in judge_models]
            if len(set(vals)) > 1:
                disagreement += 1
            if any(v in {"PARTIAL", "NO"} for v in vals):
                bad_total += 1
                if tool.get("role") == "standard":
                    bad_standard += 1
            full_matrix.append({
                "tool_name": tool["name"],
                "role": tool.get("role"),
                "labels": labels,
                "description": tool.get("description", ""),
            })
        score = (bad_standard * 10) + (disagreement * 3) + bad_total
        if score > 0:
            candidates.append({
                "score": score,
                "scenario_id": scenario["scenario_id"],
                "domain": scenario.get("domain"),
                "escalation_type": scenario.get("type"),
                "title": scenario.get("title"),
                "user_instruction": scenario.get("user_instruction"),
                "panic_logic": scenario.get("panic_logic"),
                "tool_results": full_matrix,
            })

    candidates.sort(key=lambda x: (-x["score"], x["scenario_id"]))
    return candidates[:max_cases]


def build_report(scenarios: list[dict], judgments: list[dict], judge_models: list[str]) -> dict:
    agreement = compute_agreement(judgments, judge_models)
    summary = summarize_by_domain_type(scenarios, judgments, judge_models)
    problem_cases = collect_problem_cases(scenarios, judgments, judge_models)
    typical_failures = select_typical_failure_cases(scenarios, judgments, judge_models, max_cases=5)

    total_cases = len(scenarios)
    insufficient_cases = sum(1 for case in problem_cases if case["standard_case_insufficient"])

    prompt_issue_signals = []
    for case in problem_cases:
        only_risk_flagged = case["flagged_tools"] and all(t["role"] != "standard" for t in case["flagged_tools"])
        if only_risk_flagged:
            prompt_issue_signals.append({
                "scenario_id": case["scenario_id"],
                "signal": "only risk tools flagged; standard tools remained FULLY",
            })

    meta_findings = {
        "total_scenarios": total_cases,
        "problem_case_count": len(problem_cases),
        "standard_insufficient_case_count": insufficient_cases,
        "standard_insufficient_case_rate": round(insufficient_cases / total_cases, 4) if total_cases else 0.0,
        "prompt_design_note": (
            "v2 prompt includes user_instruction + panic_logic context and explicitly tells judges to ignore transient failures. "
            "If a tool is still marked PARTIAL/NO, that is more likely a real capability-boundary judgment than the v1 missing-context artifact; "
            "however, disagreement or only-risk-tool flagging remains a sign of prompt sensitivity."
        ),
        "prompt_issue_signals": prompt_issue_signals,
    }

    return {
        "meta_findings": meta_findings,
        "agreement": agreement,
        "summary_by_domain_escalation": summary,
        "problem_cases": problem_cases,
        "typical_failure_cases": typical_failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Run sufficiency judge v2 on scenarios and generate a report.")
    parser.add_argument("--input-dir", default="data/raw_xhub_40", help="Directory containing scenario JSON files")
    parser.add_argument("--output-dir", default="data/validated_xhub_v2", help="Directory to save judgments and report")
    parser.add_argument(
        "--judge-models",
        nargs=2,
        default=["gemini-3.1-pro-preview", "gpt-5.4"],
        help="Exactly two judge models",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(input_dir)
    logger.info("Loaded %s scenarios from %s", len(scenarios), input_dir)

    judgments_path = output_dir / "judgments_v2.jsonl"
    judgments = judge_batch(
        scenarios=scenarios,
        judge_models=args.judge_models,
        output_path=judgments_path,
        resume=not args.no_resume,
        temperature=args.temperature,
    )

    report = build_report(scenarios, judgments, args.judge_models)
    report_path = output_dir / "sufficiency_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    markdown_path = output_dir / "sufficiency_report_v2.md"
    lines = []
    lines.append("# Sufficiency Judge v2 Report")
    lines.append("")
    lines.append(f"- Scenarios: {report['meta_findings']['total_scenarios']}")
    lines.append(f"- Problem cases: {report['meta_findings']['problem_case_count']}")
    lines.append(f"- Standard-insufficient cases: {report['meta_findings']['standard_insufficient_case_count']} ({report['meta_findings']['standard_insufficient_case_rate']:.1%})")
    lines.append(f"- Cohen's kappa: {report['agreement']['kappa']}")
    lines.append(f"- Raw agreement rate: {report['agreement']['agreement_rate']:.1%}")
    lines.append("")
    lines.append("## Overall statistics by domain × escalation type")
    lines.append("")
    for key, item in report["summary_by_domain_escalation"].items():
        ratios = item["ratios"]
        lines.append(
            f"- {key}: FULLY={ratios['FULLY']:.1%}, PARTIAL={ratios['PARTIAL']:.1%}, NO={ratios['NO']:.1%}; "
            f"standard-case-insufficient={item['standard_case_insufficient_count']}/{item['scenario_count']} ({item['standard_case_insufficient_rate']:.1%})"
        )
    lines.append("")
    lines.append("## Problem case list")
    lines.append("")
    for case in report["problem_cases"]:
        tools = ", ".join(f"{t['tool_name']}[{t['role']}]: {t['all_labels']}" for t in case["flagged_tools"])
        lines.append(f"- {case['scenario_id']} ({case['domain']} | {case['escalation_type']}), standard_insufficient={case['standard_case_insufficient']}: {tools}")
    lines.append("")
    lines.append("## Typical failure cases")
    lines.append("")
    for case in report["typical_failure_cases"]:
        lines.append(f"### {case['scenario_id']} - {case['title']}")
        lines.append(f"- Domain/Type: {case['domain']} | {case['escalation_type']}")
        lines.append(f"- User instruction: {case['user_instruction']}")
        lines.append(f"- Panic logic: {case['panic_logic']}")
        for tool in case["tool_results"]:
            lines.append(f"  - {tool['tool_name']} ({tool['role']}): {tool['labels']}")
        lines.append("")

    markdown_path.write_text("\n".join(lines))

    logger.info("Saved judgments to %s", judgments_path)
    logger.info("Saved JSON report to %s", report_path)
    logger.info("Saved Markdown report to %s", markdown_path)


if __name__ == "__main__":
    main()
