#!/usr/bin/env python3
"""Run the closed-loop benchmark production controller."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.closed_loop import build_controller_from_config, describe_runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Closed-loop scenario production")
    parser.add_argument("--output", default="data/closed_loop", help="Output directory for closed-loop artifacts")
    parser.add_argument("--max-total-accepted", type=int, default=None, help="Stop after this many accepted cases")
    args = parser.parse_args()

    controller = build_controller_from_config(Path(args.output))
    if args.max_total_accepted is not None:
        controller.max_total_accepted = args.max_total_accepted
    runtime = describe_runtime(controller)
    print("Closed-loop runtime:")
    print(json.dumps(runtime, indent=2, ensure_ascii=False))
    progress = controller.run()
    print(json.dumps(progress, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
