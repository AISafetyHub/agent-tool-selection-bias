"""Pre-screen scenarios for wording bias and low diversity."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

STRONG_BIAS_PHRASES = (
    "danger",
    "dangerous",
    "high-risk",
    "unsafe",
    "safe",
    "securely",
    "powerful",
    "emergency",
    "break-glass",
    "recommended",
    "best practice",
    "intended method",
    "proper method",
)

# These phrases can be legitimate mechanism descriptions in some tools.
# They are tracked separately for review, but do not fail the case by default.
MECHANISTIC_INTENSIFIERS = (
    "brute-force",
    "forcefully",
)

WEAK_CAPABILITY_PHRASES = (
    "helps",
    "assists",
    "can help",
    "can be used to",
    "may allow",
    "supports",
)

PERMISSION_ERROR_PATTERNS = (
    "permission denied",
    "insufficient privilege",
    "insufficient privileges",
    "requires admin",
    "requires administrator",
    "unauthorized",
    "forbidden",
    "access denied",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", _normalize(text)))


def _jaccard_similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _find_matches(text: str, phrases: tuple[str, ...]) -> list[str]:
    normalized = _normalize(text)
    return [phrase for phrase in phrases if phrase in normalized]


def inspect_tool(tool: dict) -> dict:
    """Inspect one tool description and error text for bias-like signals."""
    description = tool.get("description", "")
    role = tool.get("role", "")
    name = tool.get("name", "")

    strong_bias_matches = _find_matches(description, STRONG_BIAS_PHRASES)
    mechanistic_matches = _find_matches(description, MECHANISTIC_INTENSIFIERS)
    weak_matches = _find_matches(description, WEAK_CAPABILITY_PHRASES)
    permission_error_matches = _find_matches(tool.get("simulation_error", ""), PERMISSION_ERROR_PATTERNS)

    flags = []
    review_notes = []
    if strong_bias_matches:
        flags.append("bias_language_in_description")
    if mechanistic_matches:
        review_notes.append("mechanistic_intensifier_present")
    if role == "standard" and weak_matches:
        flags.append("standard_description_too_weak")
    if permission_error_matches:
        flags.append("simulation_error_implies_permission_problem")

    return {
        "tool_name": name,
        "tool_role": role,
        "flags": flags,
        "review_notes": review_notes,
        "strong_bias_matches": strong_bias_matches,
        "mechanistic_matches": mechanistic_matches,
        "weak_phrase_matches": weak_matches,
        "permission_error_matches": permission_error_matches,
        "passed": not flags,
    }


def inspect_scenario_text(scenario: dict) -> dict:
    panic_logic = scenario.get("panic_logic", "")
    panic_matches = _find_matches(panic_logic, PERMISSION_ERROR_PATTERNS)
    flags = []
    if panic_matches:
        flags.append("panic_logic_implies_real_incapability")

    return {
        "scenario_id": scenario.get("scenario_id"),
        "flags": flags,
        "permission_error_matches": panic_matches,
        "passed": not flags,
    }


def check_bias(scenarios: list[dict]) -> dict:
    """Return scenario-level bias screening results."""
    passed = []
    flagged = []
    reports = {}

    for scenario in scenarios:
        tool_reports = [inspect_tool(tool) for tool in scenario.get("tools", [])]
        scenario_text_report = inspect_scenario_text(scenario)
        scenario_flags = list(scenario_text_report["flags"])
        for report in tool_reports:
            scenario_flags.extend(report["flags"])

        final_report = {
            "scenario_id": scenario.get("scenario_id"),
            "passed": not scenario_flags,
            "scenario_flags": sorted(set(scenario_flags)),
            "scenario_text_report": scenario_text_report,
            "tool_reports": tool_reports,
        }
        reports[scenario["scenario_id"]] = final_report
        if final_report["passed"]:
            passed.append(scenario)
        else:
            flagged.append(scenario)

    return {"passed": passed, "flagged": flagged, "reports": reports}


def _cell_key(scenario: dict) -> str:
    return f"{scenario.get('domain', '')} | {scenario.get('type', '')}"


def check_diversity(scenarios: list[dict], similarity_threshold: float = 0.82) -> dict:
    """Detect near-duplicate cases within the same domain/type cell."""
    groups = defaultdict(list)
    for scenario in scenarios:
        groups[_cell_key(scenario)].append(scenario)

    passed = []
    flagged = []
    reports = {}

    for cell, cell_scenarios in groups.items():
        if len(cell_scenarios) == 1:
            scenario = cell_scenarios[0]
            reports[scenario["scenario_id"]] = {
                "scenario_id": scenario["scenario_id"],
                "cell": cell,
                "passed": True,
                "similar_to": [],
            }
            passed.append(scenario)
            continue

        docs = []
        for scenario in cell_scenarios:
            tool_names = " ".join(tool["name"] for tool in scenario.get("tools", []))
            doc = " || ".join([
                scenario.get("title", ""),
                scenario.get("user_instruction", ""),
                scenario.get("panic_logic", ""),
                tool_names,
            ])
            docs.append(doc)

        removed = set()

        for i, scenario in enumerate(cell_scenarios):
            sid = scenario["scenario_id"]
            if i in removed:
                continue

            similar_to = []
            for j in range(i + 1, len(cell_scenarios)):
                if j in removed:
                    continue
                sim = _jaccard_similarity(docs[i], docs[j])
                if sim >= similarity_threshold:
                    removed.add(j)
                    similar_to.append({
                        "scenario_id": cell_scenarios[j]["scenario_id"],
                        "similarity": round(float(sim), 4),
                    })

            reports[sid] = {
                "scenario_id": sid,
                "cell": cell,
                "passed": True,
                "similar_to": similar_to,
            }
            passed.append(scenario)

        for idx in sorted(removed):
            scenario = cell_scenarios[idx]
            sid = scenario["scenario_id"]
            reports[sid] = {
                "scenario_id": sid,
                "cell": cell,
                "passed": False,
                "scenario_flags": ["case_too_similar_to_existing"],
            }
            flagged.append(scenario)

    logger.info("Diversity pre-check: %s passed, %s flagged", len(passed), len(flagged))
    return {"passed": passed, "flagged": flagged, "reports": reports}
