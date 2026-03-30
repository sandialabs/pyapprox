#!/usr/bin/env bash
# Compare current mypy error count against a baseline.
# Usage: scripts/mypy_diff.sh [baseline_count] [directory]
# Exit code 1 if errors increased above baseline.
set -euo pipefail

BASELINE="${1:-1800}"
DIR="${2:-pyapprox/}"

OUTPUT=$(mypy "$DIR" --strict 2>&1 || true)
COUNT=$(echo "$OUTPUT" | grep -oE 'Found [0-9]+ errors?' | grep -oE '[0-9]+' || echo "0")

DIFF=$((COUNT - BASELINE))

echo "Baseline: $BASELINE"
echo "Current:  $COUNT"
echo "Delta:    $DIFF"

if [ "$COUNT" -gt "$BASELINE" ]; then
    echo "FAIL: mypy error count increased by $DIFF (from $BASELINE to $COUNT)"
    exit 1
else
    echo "OK: mypy error count did not increase (delta: $DIFF)"
fi
