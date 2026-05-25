#!/usr/bin/env python3
"""Convert raw instruction/response SFT data into conversational JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import build_sft_messages, load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    converted = []
    for idx, row in enumerate(rows):
        instruction = row["instruction"]
        response = row["response"]
        converted.append({
            "id": row.get("id", f"sft-{idx:07d}"),
            "messages": build_sft_messages(instruction, response),
            "instruction": instruction,
            "response": response,
        })

    write_jsonl(args.output, converted)
    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "count": len(converted),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
