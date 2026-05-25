"""Two-judge sufficiency verification."""

import json
import logging
from pathlib import Path

import yaml

from ..api_client import chat_completion, get_client

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (Path(__file__).parent.parent.parent / "prompts" / "sufficiency_judge.txt").read_text()
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def _load_judge_config() -> dict:
    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        return yaml.safe_load(f)["validation"]


def judge_tool(
    user_instruction: str,
    tool: dict,
    model: str,
    client=None,
) -> str:
    """Query one judge model for a single (task, tool) pair. Returns FULLY/PARTIAL/NO."""
    client = client or get_client()
    prompt = PROMPT_TEMPLATE.format(
        user_instruction=user_instruction,
        tool_name=tool["name"],
        tool_description=tool.get("description", ""),
        tool_parameters=json.dumps(tool.get("parameters", {})),
    )

    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=16,
        client=client,
    )

    text = (response.choices[0].message.content or "").strip().upper()
    # Normalize: extract first valid label
    for label in ("FULLY", "PARTIAL", "NO"):
        if label in text:
            return label
    logger.warning(f"Unexpected judge response: '{text}' for tool={tool['name']}, model={model}")
    return "NO"


def judge_scenario(
    scenario: dict,
    judge_models: list[str],
    client=None,
) -> dict:
    """Judge all tools in a scenario with all judge models.

    Returns {tool_name: {model_id: label}}.
    """
    client = client or get_client()
    results = {}
    for tool in scenario["tools"]:
        results[tool["name"]] = {}
        for model in judge_models:
            label = judge_tool(
                user_instruction=scenario["user_instruction"],
                tool=tool,
                model=model,
                client=client,
            )
            results[tool["name"]][model] = label
    return results


def judge_batch(
    scenarios: list[dict],
    judge_models: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """Judge all scenarios. Save intermediate results for resume support.

    Returns list of {scenario_id, judgments: {tool_name: {model: label}}}.
    """
    cfg = _load_judge_config()
    judge_models = judge_models or cfg["judge_models"]
    client = get_client()

    # Load existing results for resume
    existing = {}
    if output_path and output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["scenario_id"]] = rec

    results = []
    fh = open(output_path, "a") if output_path else None

    try:
        for i, scenario in enumerate(scenarios):
            sid = scenario["scenario_id"]
            if sid in existing:
                results.append(existing[sid])
                continue

            logger.info(f"[{i+1}/{len(scenarios)}] Judging {sid}")
            judgments = judge_scenario(scenario, judge_models, client=client)
            rec = {"scenario_id": sid, "judgments": judgments}
            results.append(rec)

            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()

    logger.info(f"Judged {len(results)} scenarios")
    return results
