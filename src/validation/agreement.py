"""Compute inter-judge agreement and filter benchmark."""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def compute_agreement(judgments: list[dict], judge_models: list[str]) -> dict:
    """Compute Cohen's kappa between two judges across all tool-task pairs.

    judgments: list of {scenario_id, judgments: {tool_name: {model: label}}}
    """
    if len(judge_models) != 2:
        raise ValueError("Exactly 2 judge models required for Cohen's kappa")

    m1, m2 = judge_models
    labels_1, labels_2 = [], []

    for rec in judgments:
        for tool_name, model_labels in rec["judgments"].items():
            if m1 in model_labels and m2 in model_labels:
                labels_1.append(model_labels[m1])
                labels_2.append(model_labels[m2])

    total = len(labels_1)
    agree = sum(a == b for a, b in zip(labels_1, labels_2))
    p0 = agree / total if total > 0 else 0.0
    if total > 0:
        c1 = Counter(labels_1)
        c2 = Counter(labels_2)
        pe = sum((c1[label] / total) * (c2[label] / total) for label in set(labels_1) | set(labels_2))
        kappa = (p0 - pe) / (1 - pe) if pe != 1.0 else 0.0
    else:
        kappa = 0.0

    return {
        "total_pairs": total,
        "agreement_count": agree,
        "agreement_rate": p0,
        "kappa": round(kappa, 4),
    }


def filter_benchmark(
    scenarios: list[dict],
    judgments: list[dict],
    judge_models: list[str],
    require_both_fully: bool = True,
) -> tuple[list[dict], dict]:
    """Keep only scenarios where both judges rate ALL standard tools as FULLY.

    Returns (filtered_scenarios, stats).
    """
    judgment_map = {rec["scenario_id"]: rec["judgments"] for rec in judgments}

    kept = []
    removed_reasons = {"missing_judgment": 0, "not_fully": 0}

    for s in scenarios:
        sid = s["scenario_id"]
        if sid not in judgment_map:
            removed_reasons["missing_judgment"] += 1
            continue

        j = judgment_map[sid]
        standard_tools = [t for t in s["tools"] if t["role"] == "standard"]

        all_fully = True
        for tool in standard_tools:
            tool_judgments = j.get(tool["name"], {})
            for model in judge_models:
                if tool_judgments.get(model) != "FULLY":
                    all_fully = False
                    break
            if not all_fully:
                break

        if all_fully or not require_both_fully:
            kept.append(s)
        else:
            removed_reasons["not_fully"] += 1

    stats = {
        "initial_count": len(scenarios),
        "filtered_count": len(kept),
        "retention_rate": round(len(kept) / len(scenarios), 4) if scenarios else 0.0,
        "removed": removed_reasons,
    }
    logger.info(f"Filtering: {stats}")
    return kept, stats
