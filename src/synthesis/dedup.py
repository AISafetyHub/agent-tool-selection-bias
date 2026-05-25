"""Deduplication and quality filtering for generated scenarios."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def structural_dedup(scenarios: list[dict]) -> list[dict]:
    """Remove scenarios whose tool names collide with earlier ones."""
    seen_names: set[str] = set()
    kept = []
    for s in scenarios:
        names = {t["name"] for t in s["tools"]}
        if names & seen_names:
            logger.debug(f"Removing {s['scenario_id']}: duplicate tool names")
            continue
        seen_names |= names
        kept.append(s)
    logger.info(f"Structural dedup: {len(scenarios)} -> {len(kept)}")
    return kept


def content_dedup(scenarios: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove scenarios with highly similar user_instructions via token Jaccard similarity."""
    if len(scenarios) <= 1:
        return scenarios

    # Greedy removal: for each pair above threshold, remove the later one
    remove_indices: set[int] = set()
    for i in range(len(scenarios)):
        if i in remove_indices:
            continue
        for j in range(i + 1, len(scenarios)):
            if j in remove_indices:
                continue
            if _jaccard_similarity(scenarios[i]["user_instruction"], scenarios[j]["user_instruction"]) > threshold:
                remove_indices.add(j)

    kept = [s for i, s in enumerate(scenarios) if i not in remove_indices]
    logger.info(f"Content dedup (threshold={threshold}): {len(scenarios)} -> {len(kept)}")
    return kept


def filter_and_dedup(
    input_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.85,
) -> dict:
    """Full pipeline: load -> structural dedup -> content dedup -> save."""
    scenarios = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    initial = len(scenarios)
    scenarios = structural_dedup(scenarios)
    scenarios = content_dedup(scenarios, threshold=similarity_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "initial_count": initial,
        "after_dedup": len(scenarios),
        "removed": initial - len(scenarios),
    }
    logger.info(f"Dedup complete: {stats}")
    return stats
