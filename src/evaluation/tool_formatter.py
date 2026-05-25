"""Format scenario tools as OpenAI function-calling schemas."""

import copy
import random


def format_tools(tools: list[dict], randomize_order: bool = True) -> tuple[list[dict], list[str]]:
    """Convert scenario tools to OpenAI function-calling tool schema format.

    Returns:
        (formatted_tools, order): order is the list of tool names in presentation order.
    """
    formatted = []
    for t in tools:
        params = t.get("parameters", {})
        # Ensure valid JSON Schema structure
        if params and "type" not in params:
            params = {"type": "object", "properties": params}
        elif not params:
            params = {"type": "object", "properties": {}}

        formatted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": params,
            },
        })

    if randomize_order:
        random.shuffle(formatted)

    order = [f["function"]["name"] for f in formatted]
    return formatted, order


def build_tool_lookup(tools: list[dict]) -> dict[str, dict]:
    """Build name -> tool dict for quick role/output lookup during simulation."""
    return {t["name"]: t for t in tools}
