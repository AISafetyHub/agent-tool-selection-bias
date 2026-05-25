#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Step 1: Synthesize scenarios ==="
python scripts/01_synthesize.py "$@"

echo ""
echo "=== Step 2: Validate & filter ==="
python scripts/02_validate.py

echo ""
echo "=== Step 3: Evaluate models ==="
python scripts/03_evaluate.py

echo ""
echo "=== Step 4: Analyze results ==="
python scripts/04_analyze.py

echo ""
echo "Done. Results in data/results/"
