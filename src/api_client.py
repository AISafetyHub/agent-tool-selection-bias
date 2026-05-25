"""Unified API client wrapper with OpenAI- and xhub-compatible helpers."""

import json
import os
import time
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import requests
import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pipeline.yaml"


def _resolve_api_key(preferred_envs: list[str]) -> str:
    for env_name in preferred_envs:
        value = os.environ.get(env_name)
        if value:
            return value
    tried = ", ".join(preferred_envs)
    raise KeyError(f"No API key found in environment variables: {tried}")


def _load_api_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["api"]


def get_api_config() -> dict:
    """Expose loaded API config for callers that want explicit runtime checks/logging."""
    return _load_api_config()


def _is_xhub_base_url(base_url: str) -> bool:
    return "api3.xhub.chat" in (base_url or "")


def _normalize_xhub_base_url(base_url: str) -> str:
    return (base_url or "https://api3.xhub.chat").rstrip("/")


def _normalize_xhub_model(model: str) -> str:
    if "/" in (model or ""):
        return model.split("/", 1)[1]
    return model


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI-compatible client for standard chat-completions backends."""
    cfg = _load_api_config()
    return OpenAI(
        api_key=api_key or _resolve_api_key(["OPENROUTER_API_KEY", "XHUB_API_KEY"]),
        base_url=cfg["base_url"],
    )


def get_default_client(api_key: Optional[str] = None):
    """Return the default client matching the configured backend."""
    cfg = _load_api_config()
    if _is_xhub_base_url(cfg.get("base_url", "")):
        return get_xhub_client(api_key=api_key)
    return get_client(api_key=api_key)


def get_xhub_client(api_key: Optional[str] = None) -> dict:
    """Return lightweight config for the xhub Anthropic-compatible /v1/messages API."""
    cfg = _load_api_config()
    return {
        "api_key": api_key or _resolve_api_key(["XHUB_API_KEY", "OPENROUTER_API_KEY"]),
        "base_url": _normalize_xhub_base_url(cfg.get("base_url", "https://api3.xhub.chat")),
        "timeout": cfg.get("timeout", 120),
        "max_retries": cfg.get("max_retries", 3),
        "retry_delay": cfg.get("retry_delay", 2.0),
    }


def require_xhub_config() -> dict:
    """Assert that the configured API backend is xhub and return normalized config."""
    cfg = get_api_config()
    base_url = cfg.get("base_url", "")
    if not _is_xhub_base_url(base_url):
        raise RuntimeError(
            f"Closed-loop data production is configured to require xhub, but api.base_url={base_url!r}"
        )
    normalized = dict(cfg)
    normalized["base_url"] = _normalize_xhub_base_url(base_url)
    return normalized


def _split_system_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    system_parts = []
    converted = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        system_parts.append(block)
            else:
                system_parts.append(content or "")
            continue

        if role == "assistant" and msg.get("tool_calls"):
            blocks = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for tool_call in msg.get("tool_calls", []):
                fn = tool_call.get("function", {})
                arguments = fn.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": arguments}
                blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": fn.get("name"),
                    "input": arguments,
                })
            converted.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": content or "",
                }],
            })
            continue

        if isinstance(content, str):
            converted_content = content
        else:
            converted_content = content
        converted.append({"role": role, "content": converted_content})

    system_text = "\n\n".join(p for p in system_parts if p)
    return (system_text or None), converted


def _extract_xhub_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _extract_xhub_tool_calls(content) -> list[SimpleNamespace] | None:
    if not isinstance(content, list):
        return None
    tool_calls = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        tool_calls.append(SimpleNamespace(
            id=block.get("id"),
            function=SimpleNamespace(
                name=block.get("name"),
                arguments=json.dumps(block.get("input", {}), ensure_ascii=False),
            ),
        ))
    return tool_calls or None


def _xhub_to_openai_response(payload: dict):
    text = _extract_xhub_text_content(payload.get("content", []))
    tool_calls = _extract_xhub_tool_calls(payload.get("content", []))
    usage_raw = payload.get("usage", {}) or {}
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=usage_raw.get("input_tokens", 0),
        completion_tokens=usage_raw.get("output_tokens", 0),
        total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
    )
    return SimpleNamespace(
        id=payload.get("id"),
        model=payload.get("model"),
        object="chat.completion",
        choices=[choice],
        usage=usage,
        raw_response=payload,
    )


def chat_completion_xhub(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.9,
    max_tokens: int = 8192,
    api_key: str = None,
    base_url: str = "https://api3.xhub.chat",
    timeout: int = 120,
) -> dict:
    """Call xhub /v1/messages endpoint (Anthropic-compatible)."""
    url = f"{_normalize_xhub_base_url(base_url)}/v1/messages"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    system_text, anthropic_messages = _split_system_messages(messages)
    payload = {
        "model": _normalize_xhub_model(model),
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_text:
        payload["system"] = system_text
    if tools:
        payload["tools"] = [{
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
        } for tool in tools]

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def chat_completion(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    client: Optional[OpenAI] = None,
) -> dict:
    """Send a chat completion request with retry logic.

    Returns an OpenAI-SDK response object, or an OpenAI-like shim for xhub.
    """
    cfg = _load_api_config()
    base_url = cfg.get("base_url", "")
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_delay", 2.0)

    if _is_xhub_base_url(base_url):
        xhub_client = client or get_xhub_client()
        for attempt in range(max_retries):
            try:
                payload = chat_completion_xhub(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=xhub_client["api_key"],
                    base_url=xhub_client["base_url"],
                    timeout=xhub_client.get("timeout", cfg.get("timeout", 120)),
                )
                return _xhub_to_openai_response(payload)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"xhub API call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s")
                time.sleep(wait)

    client = client or get_client()
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = retry_delay * (2 ** attempt)
            logger.warning(f"API call failed (attempt {attempt + 1}): {e}. Retrying in {wait}s")
            time.sleep(wait)


def extract_tool_call(response) -> Optional[dict]:
    """Extract the first tool call from a chat completion response.

    Returns dict with keys: name, arguments (parsed dict), or None if no tool call.
    """
    message = response.choices[0].message
    if not getattr(message, "tool_calls", None):
        return None
    tc = message.tool_calls[0]
    return {
        "id": tc.id,
        "name": tc.function.name,
        "arguments": json.loads(tc.function.arguments),
    }


def extract_text(response) -> str:
    """Extract text content from a chat completion response."""
    return response.choices[0].message.content or ""


def extract_usage(response) -> dict:
    """Extract normalized token usage from an OpenAI-like response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def response_to_dict(response) -> dict:
    """Best-effort serialization of a response object for trajectory logging."""
    raw = getattr(response, "raw_response", None)
    if isinstance(raw, dict):
        return raw
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    message = response.choices[0].message
    tool_calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        tool_calls.append({
            "id": getattr(tc, "id", None),
            "function": {
                "name": getattr(getattr(tc, "function", None), "name", None),
                "arguments": getattr(getattr(tc, "function", None), "arguments", None),
            },
        })
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "content": getattr(message, "content", None),
        "tool_calls": tool_calls,
        "usage": extract_usage(response),
    }
