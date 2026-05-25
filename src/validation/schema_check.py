"""Validate scenarios against JSON Schema."""

import json
import logging
from pathlib import Path

try:
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    jsonschema = None

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "scenario.json"


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_scenario(scenario: dict, schema: dict | None = None) -> tuple[bool, list[str]]:
    """Validate a single scenario. Returns (is_valid, list_of_errors)."""
    schema = schema or load_schema()
    errors = []

    # JSON Schema validation
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(scenario):
            errors.append(f"Schema: {err.message} at {list(err.absolute_path)}")
    else:
        logger.warning("jsonschema not installed; falling back to structural validation only")

    # Structural checks beyond JSON Schema
    tools = scenario.get("tools", [])
    standard = [t for t in tools if t.get("role") == "standard"]
    risk = [t for t in tools if t.get("role") == "risk"]

    if len(standard) != 3:
        errors.append(f"Expected 3 standard tools, got {len(standard)}")
    if len(risk) != 3:
        errors.append(f"Expected 3 risk tools, got {len(risk)}")

    # Standard tools must have non-empty simulation_error
    for t in standard:
        if not t.get("simulation_error"):
            errors.append(f"Standard tool '{t.get('name')}' missing simulation_error")

    # Risk tools should have empty simulation_error
    for t in risk:
        if t.get("simulation_error"):
            errors.append(f"Risk tool '{t.get('name')}' should not have simulation_error")

    # Tool names must be unique within scenario
    names = [t.get("name") for t in tools]
    if len(names) != len(set(names)):
        errors.append("Duplicate tool names within scenario")

    return len(errors) == 0, errors


def validate_batch(scenarios: list[dict]) -> dict:
    """Validate all scenarios. Returns categorized results."""
    schema = load_schema()
    valid = []
    invalid = []
    all_errors = {}

    for s in scenarios:
        sid = s.get("scenario_id", "unknown")
        ok, errs = validate_scenario(s, schema)
        if ok:
            valid.append(s)
        else:
            invalid.append(s)
            all_errors[sid] = errs

    logger.info(f"Schema validation: {len(valid)} valid, {len(invalid)} invalid")
    return {"valid": valid, "invalid": invalid, "errors": all_errors}
