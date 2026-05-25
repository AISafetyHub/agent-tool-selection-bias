"""Build repair plans and prompts for failed scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import logging

from ..api_client import chat_completion, extract_text, extract_usage, get_xhub_client

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
REPAIR_TEMPLATE = (ROOT / "prompts" / "repair_case.txt").read_text()


def _collect_banned_phrases(bias_report: dict) -> dict[str, list[str]]:
    phrases = {}
    for tool_report in bias_report.get("tool_reports", []):
        matches = tool_report.get("strong_bias_matches", [])
        if matches:
            phrases[tool_report["tool_name"]] = sorted(set(matches))
    return phrases


def build_repair_prompt(
    scenario: dict,
    bias_report: dict | None = None,
    failure_record: dict | None = None,
    diversity_report: dict | None = None,
) -> str:
    """Render one targeted repair prompt from validation artifacts."""
    bias_report = bias_report or {}
    failure_record = failure_record or {}
    diversity_report = diversity_report or {}

    scenario_tags = set(bias_report.get("scenario_flags", []))
    scenario_tags.update(failure_record.get("scenario_tags", []))
    scenario_tags.update(diversity_report.get("scenario_flags", []))

    similar_cases = diversity_report.get("similar_to", [])
    banned_phrases = _collect_banned_phrases(bias_report)
    tool_failures = failure_record.get("tool_failures", [])

    return REPAIR_TEMPLATE.format(
        original_case_json=json.dumps(scenario, ensure_ascii=False, indent=2),
        scenario_tags=json.dumps(sorted(scenario_tags), ensure_ascii=False),
        tool_failures_json=json.dumps(tool_failures, ensure_ascii=False, indent=2),
        banned_phrases_json=json.dumps(banned_phrases, ensure_ascii=False, indent=2),
        similar_cases_json=json.dumps(similar_cases, ensure_ascii=False, indent=2),
    )


def build_repair_candidates(
    scenarios: list[dict],
    bias_reports: dict[str, dict],
    failure_analysis: list[dict] | None = None,
    diversity_reports: dict[str, dict] | None = None,
) -> list[dict]:
    """Create structured repair candidates from validation artifacts."""
    failure_map = {record["scenario_id"]: record for record in (failure_analysis or [])}
    diversity_reports = diversity_reports or {}

    candidates = []
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        bias_report = bias_reports.get(sid, {})
        failure_record = failure_map.get(sid, {})
        diversity_report = diversity_reports.get(sid, {})

        scenario_tags = set(bias_report.get("scenario_flags", []))
        scenario_tags.update(failure_record.get("scenario_tags", []))
        scenario_tags.update(diversity_report.get("scenario_flags", []))

        if not scenario_tags:
            continue

        strategy = "discard_and_regenerate"
        if "case_may_be_repaired" in scenario_tags or "bias_language_in_description" in scenario_tags:
            strategy = "targeted_repair"
        if "case_should_be_regenerated" in scenario_tags or "case_too_similar_to_existing" in scenario_tags:
            strategy = "discard_and_regenerate"

        candidates.append({
            "scenario_id": sid,
            "domain": scenario.get("domain"),
            "type": scenario.get("type"),
            "strategy": strategy,
            "scenario_tags": sorted(scenario_tags),
            "repair_prompt": build_repair_prompt(
                scenario,
                bias_report=bias_report,
                failure_record=failure_record,
                diversity_report=diversity_report,
            ),
        })

    return candidates


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_repaired_case(text: str) -> dict:
    """Parse one repaired case JSON object from model output."""
    cleaned = _strip_fences(text)
    payload = json.loads(cleaned)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("Repair model returned a JSON array with unexpected length")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Repair model did not return a JSON object")
    return payload


def repair_case(
    prompt: str,
    model: str,
    client=None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    include_meta: bool = False,
) -> dict | tuple[dict, dict]:
    """Call the repair model and parse the revised scenario JSON."""
    client = client or get_xhub_client()
    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )
    text = extract_text(response)
    usage = extract_usage(response)
    try:
        repaired = parse_repaired_case(text)
        if include_meta:
            return repaired, {"model": model, "usage": usage}
        return repaired
    except Exception as e:
        logger.error("Failed to parse repaired case: %s", e)
        logger.debug("Raw repair response: %s", text[:1000])
        raise
