"""Aggregate metrics by model, domain, escalation type."""

import json
import logging
from pathlib import Path
from collections import defaultdict

from .opur import compute_opur_curve, compute_over_privilege_curve
from .ped import compute_ped

logger = logging.getLogger(__name__)


def load_logs(eval_logs_dir: Path) -> dict[str, list[dict]]:
    """Load all evaluation logs grouped by model.

    Returns {model_name: [log_dict, ...]}.
    """
    model_logs = {}
    for model_dir in eval_logs_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        logs = []
        for f in sorted(model_dir.glob("*.json")):
            with open(f) as fh:
                logs.append(json.load(fh))
        if logs:
            model_logs[model_name] = logs
    return model_logs


def aggregate(eval_logs_dir: Path, k_values: list[int] = None) -> dict:
    """Compute all metrics aggregated across multiple dimensions.

    Returns nested dict:
      overall: {model: {opur@k, ped}}
      by_domain: {model: {domain: {opur@k, ped}}}
      by_type: {model: {type: {opur@k, ped}}}
    """
    k_values = k_values or [1, 3, 5]
    model_logs = load_logs(eval_logs_dir)

    results = {"overall": {}, "by_domain": {}, "by_type": {}}

    for model, logs in model_logs.items():
        # Overall
        results["overall"][model] = {
            "opur": compute_opur_curve(logs, k_values),
            "over_privilege_rate": compute_over_privilege_curve(logs, k_values),
            "ped": compute_ped(logs),
            "total_scenarios": len(logs),
        }

        # By domain
        domain_groups = defaultdict(list)
        for log in logs:
            domain_groups[log.get("domain", "unknown")].append(log)
        results["by_domain"][model] = {
            domain: {
                "opur": compute_opur_curve(dlogs, k_values),
                "over_privilege_rate": compute_over_privilege_curve(dlogs, k_values),
                "ped": compute_ped(dlogs),
            }
            for domain, dlogs in domain_groups.items()
        }

        # By escalation type
        type_groups = defaultdict(list)
        for log in logs:
            type_groups[log.get("type", "unknown")].append(log)
        results["by_type"][model] = {
            etype: {
                "opur": compute_opur_curve(tlogs, k_values),
                "over_privilege_rate": compute_over_privilege_curve(tlogs, k_values),
                "ped": compute_ped(tlogs),
            }
            for etype, tlogs in type_groups.items()
        }

    return results
