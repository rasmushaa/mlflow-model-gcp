#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Compile full project requirements ==="
uv pip compile pyproject.toml -o requirements.txt > /dev/null 2>&1

echo "=== Step 2: Extract model-only requirements (unpinned) ==="
uv run pipreqs src/models --force --encoding=utf-8 --mode no-pin > /dev/null 2>&1

echo "=== Step 3: Pin model requirements based on global uv versions ==="
uv run python scripts/pin_model_reqs.py > /dev/null 2>&1

echo "=== Done: Model requirements built ==="
