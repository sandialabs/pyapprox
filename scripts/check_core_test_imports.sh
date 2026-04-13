#!/usr/bin/env bash
# Verify that packages/pyapprox/tests/ has no pyapprox_benchmarks imports.
# This enforces the package boundary that breaks the install cycle.
set -euo pipefail

matches=$(grep -r 'import pyapprox_benchmarks\|from pyapprox_benchmarks' packages/pyapprox/tests/ 2>/dev/null || true)
if [ -n "$matches" ]; then
    echo "ERROR: pyapprox core tests must not import pyapprox_benchmarks."
    echo "Move these tests to tests/integration/ instead."
    echo ""
    echo "$matches"
    exit 1
fi
echo "OK: no pyapprox_benchmarks imports in packages/pyapprox/tests/"
