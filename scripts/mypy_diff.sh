#!/usr/bin/env bash
# Compare current mypy error counts against their baselines.
# Usage: scripts/mypy_diff.sh [baseline_count] [directory]
#
# Two counts are checked. In-place writes into a backend array
# ("Unsupported target for indexed assignment": ArrayProtocol has no
# __setitem__) are counted on their own against the shrink-only
# inplace_writes:<package> entry of ratchet_baselines.txt, where
# <package> is the last component of the directory (pyapprox,
# pyapprox_benchmarks), and only for files inside the directory; every
# other error counts against baseline_count. Pass the directory as CI does, e.g.
# packages/pyapprox/src/pyapprox/.
# Exit code 1 if either count rose above its baseline.
set -euo pipefail

BASELINE="${1:-1325}"
DIR="${2:-packages/pyapprox/src/pyapprox/}"
INPLACE_MESSAGE="Unsupported target for indexed assignment"
PACKAGE="$(basename "${DIR%/}")"
BASELINES_FILE="$(dirname "$0")/../ratchet_baselines.txt"

OUTPUT=$(mypy "$DIR" --strict 2>&1 || true)
ERRORS=$(echo "$OUTPUT" | grep -E ': error: ' || true)
TOTAL=$(echo "$ERRORS" | grep -c ': error: ' || true)
# mypy also reports modules it follows outside DIR (pyapprox from the
# benchmarks): every in-place error leaves the general count, but only
# those inside DIR count against this package's in-place baseline.
INPLACE_ALL=$(echo "$ERRORS" | grep -c "$INPLACE_MESSAGE" || true)
INPLACE=$(echo "$ERRORS" | grep "$INPLACE_MESSAGE" | grep -c "^${DIR%/}/" || true)
COUNT=$((TOTAL - INPLACE_ALL))
INPLACE_BASELINE=$(sed -n "s/^inplace_writes:${PACKAGE}=//p" "$BASELINES_FILE")
if [ -z "$INPLACE_BASELINE" ]; then
    INPLACE_BASELINE=0
fi

DIFF=$((COUNT - BASELINE))
INPLACE_DIFF=$((INPLACE - INPLACE_BASELINE))

echo "Baseline: $BASELINE"
echo "Current:  $COUNT"
echo "Delta:    $DIFF"
echo "In-place writes (inplace_writes:${PACKAGE}): $INPLACE (baseline $INPLACE_BASELINE)"

FAILED=0
if [ "$COUNT" -gt "$BASELINE" ]; then
    echo "FAIL: mypy error count increased by $DIFF (from $BASELINE to $COUNT)"
    FAILED=1
fi
if [ "$INPLACE" -gt "$INPLACE_BASELINE" ]; then
    echo "FAIL: in-place writes into backend arrays increased by $INPLACE_DIFF;" \
        "use bkd.index_update / bkd.index_add or collect and stack instead"
    FAILED=1
elif [ "$INPLACE" -lt "$INPLACE_BASELINE" ]; then
    echo "NOTE: in-place writes fell to $INPLACE; set inplace_writes:${PACKAGE}=$INPLACE"
fi
if [ "$FAILED" -ne 0 ]; then
    echo "----- full mypy output -----"
    echo "$OUTPUT"
    exit 1
fi
echo "OK: mypy error count did not increase (delta: $DIFF)"
