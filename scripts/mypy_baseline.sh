#!/usr/bin/env bash
# Run mypy and report the error count.
# Usage: scripts/mypy_baseline.sh [directory]
set -euo pipefail

DIR="${1:-pyapprox/}"
OUTPUT=$(mypy "$DIR" --strict 2>&1 || true)
COUNT=$(echo "$OUTPUT" | grep -oE 'Found [0-9]+ errors?' | grep -oE '[0-9]+' || echo "0")
echo "$OUTPUT"
echo ""
echo "=== MYPY ERROR COUNT: $COUNT ==="
