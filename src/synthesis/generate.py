"""Batch scenario generation using configurable LLM."""

import json
import logging
from pathlib import Path

import yaml

from ..api_client import chat_completion, extract_text, extract_usage, get_client, get_xhub_client

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (Path(__file__).parent.parent.parent / "prompts" / "synthesis.txt").read_text()
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

DOMAIN_SHORT = {
    "Infrastructure": "INFRA", "Business": "BIZ", "Coding": "CODE",
    "Media": "MEDIA", "Education": "EDU", "Government": "GOV",
    "Database": "DB", "Healthcare": "HEALTH",
}
TYPE_SHORT = {
    "Authority Escalation": "AUTH", "Scope Expansion": "SCOPE",
    "Temporal Persistence": "TEMP", "Safety Bypass": "SAFETY",
    "Data Over-Exposure": "DATA",
}


def load_domains() -> list[dict]:
    with open(CONFIGS_DIR / "domains.yaml") as f:
        return yaml.safe_load(f)["domains"]


def load_escalation_types() -> list[dict]:
    with open(CONFIGS_DIR / "escalation_types.yaml") as f:
        return yaml.safe_load(f)["escalation_types"]


def generate_batch(
    domain: dict,
    escalation_type: dict,
    count: int,
    model: str,
    client=None,
    include_meta: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Generate `count` scenarios for one (domain, escalation_type) pair."""
    client = client or get_xhub_client()
    prompt = PROMPT_TEMPLATE.format(
        count=count,
        domain=domain["name"],
        domain_short=DOMAIN_SHORT[domain["name"]],
        escalation_type=escalation_type["name"],
        type_short=TYPE_SHORT[escalation_type["name"]],
        escalation_definition=escalation_type["definition"].strip(),
    )

    response = chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=8192,
        client=client,
    )
    usage = extract_usage(response)

    text = extract_text(response)
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON for {domain['name']} x {escalation_type['name']}")
        logger.debug(f"Raw response: {text[:500]}")
        return []

    if not isinstance(scenarios, list):
        scenarios = [scenarios]

    # Assign proper scenario_ids
    ds = DOMAIN_SHORT[domain["name"]]
    ts = TYPE_SHORT[escalation_type["name"]]
    for i, s in enumerate(scenarios):
        s["scenario_id"] = f"{ds}-{ts}-{i + 1:03d}"
        s["type"] = escalation_type["name"]
        s["domain"] = domain["name"]

    if include_meta:
        return scenarios, {
            "model": model,
            "usage": usage,
            "count_requested": count,
            "count_returned": len(scenarios),
        }
    return scenarios


def generate_all(
    total: int,
    model: str,
    output_dir: Path,
) -> list[dict]:
    """Distribute total across 8x5=40 (domain, type) cells, generate, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = load_domains()
    escalation_types = load_escalation_types()

    cells = [(d, e) for d in domains for e in escalation_types]
    per_cell = total // len(cells)
    remainder = total % len(cells)

    client = get_xhub_client()
    all_scenarios = []
    global_idx = 0

    for idx, (domain, etype) in enumerate(cells):
        count = per_cell + (1 if idx < remainder else 0)
        if count == 0:
            continue

        label = f"{domain['name']} x {etype['name']}"
        logger.info(f"Generating {count} scenarios for {label}")

        batch = generate_batch(domain, etype, count, model, client=client)
        logger.info(f"  Got {len(batch)} scenarios for {label}")

        # Re-index globally to avoid id collisions across batches
        ds = DOMAIN_SHORT[domain["name"]]
        ts = TYPE_SHORT[etype["name"]]
        for s in batch:
            global_idx += 1
            s["scenario_id"] = f"{ds}-{ts}-{global_idx:03d}"
            all_scenarios.append(s)

    # Save individual JSONs
    for s in all_scenarios:
        path = output_dir / f"{s['scenario_id']}.json"
        with open(path, "w") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)

    # Save combined JSONL
    jsonl_path = output_dir / "scenarios_raw.jsonl"
    with open(jsonl_path, "w") as f:
        for s in all_scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(all_scenarios)} total scenarios -> {output_dir}")
    return all_scenarios
