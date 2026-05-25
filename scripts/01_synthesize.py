#!/usr/bin/env python3
"""Step 1: Generate evaluation scenarios."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.synthesis.generate import generate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation scenarios")
    parser.add_argument("--model", default="openai/gpt-5.2", help="Generation model")
    parser.add_argument("--total", type=int, default=400, help="Total scenarios to generate")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    scenarios = generate_all(total=args.total, model=args.model, output_dir=output_dir)
    print(f"Generated {len(scenarios)} scenarios -> {output_dir}")


if __name__ == "__main__":
    main()
