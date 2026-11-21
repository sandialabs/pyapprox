"""Keeping calls to superseded backend methods from spreading.

:data:`SUPERSEDED` names the ``Backend`` methods kept only until their
callers are converted, each with its replacement. :func:`count_calls`
counts calls to each (``.name(``, so definitions do not count) in the
repository's Python and Quarto sources. The counts are recorded in
``ratchet_baselines.txt`` as ``superseded-method:<name>=<count>``, a key
distinct from the ``superseded:`` package baselines of
:mod:`pyapprox._moved_ratchet`, since each ratchet reports a key outside
its own table as stale. No commit raises a baseline. A method is deleted
from the backends when its count reaches zero.

``python -m pyapprox._superseded_ratchet`` exits nonzero when a count
exceeds its baseline, a method has none, or a baseline is stale. A count
below its baseline is printed as a notice to lower it and does not fail,
since merging branches lowers counts no single branch could record.
"""

import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

from pyapprox._moved_ratchet import BASELINE_FILE, _files

SUPERSEDED: Mapping[str, str] = {
    "any_bool": "bkd.to_bool(bkd.any(x))",
    "all_bool": "bkd.to_bool(bkd.all(x))",
    "any_array": "bkd.any(x, axis, keepdims)",
    "all_array": "bkd.all(x, axis, keepdims)",
}
CALL_SUFFIXES = (".py", ".qmd")
BASELINE_KEY = "superseded-method:"
# Its fixtures write calls as text, so they never reach zero.
EXEMPT = ("tests/test_superseded_ratchet.py",)


def count_calls(names: Sequence[str], root: Path) -> dict[str, int]:
    """Calls ``.name(`` to each of ``names`` under ``root``."""
    counts = dict.fromkeys(names, 0)
    if not names:
        return counts
    pattern = re.compile(rf"\.({'|'.join(re.escape(n) for n in names)})\(")
    for path in _files(root):
        if path.suffix not in CALL_SUFFIXES or path.as_posix().endswith(EXEMPT):
            continue
        for match in pattern.finditer(path.read_text(errors="ignore")):
            counts[match.group(1)] += 1
    return counts


def read_baseline(path: Path) -> dict[str, int]:
    """The ``superseded-method:<name>=<count>`` entries of a baseline file."""
    baseline: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith(BASELINE_KEY) and "=" in line:
            key, value = line[len(BASELINE_KEY) :].split("=", 1)
            baseline[key.strip()] = int(value)
    return baseline


def ratchet_problems(
    table: Mapping[str, str], counts: Mapping[str, int], baseline: Mapping[str, int]
) -> list[str]:
    """What fails the ratchet: a count above its baseline, a method
    without one, and baselines of methods no longer superseded. A count
    below its baseline is a notice (:func:`ratchet_notices`)."""
    problems = []
    for name, replacement in table.items():
        count, key = counts.get(name, 0), f"{BASELINE_KEY}{name}"
        if name not in baseline:
            problems.append(f"no baseline for {name}: add {key}={count}")
        elif count > baseline[name]:
            problems.append(
                f"{name}: {count} calls, baseline {baseline[name]}; "
                f"use {replacement} instead"
            )
    problems += [
        f"{BASELINE_KEY}{n} is stale: {n} is not superseded"
        for n in baseline
        if n not in table
    ]
    return problems


def ratchet_notices(
    table: Mapping[str, str], counts: Mapping[str, int], baseline: Mapping[str, int]
) -> list[str]:
    """Counts below their baselines, each with the baseline to set."""
    return [
        f"{name}: down to {counts.get(name, 0)}; "
        f"set {BASELINE_KEY}{name}={counts.get(name, 0)}"
        for name in table
        if name in baseline and counts.get(name, 0) < baseline[name]
    ]


def main(argv: Sequence[str]) -> int:
    root = Path(__file__).resolve().parents[4]
    counts = count_calls(list(SUPERSEDED), root)
    baseline = read_baseline(root / BASELINE_FILE)
    problems = ratchet_problems(SUPERSEDED, counts, baseline)
    for problem in problems:
        print(f"superseded-method ratchet: {problem}")
    # GitHub Actions shows a ::notice:: line as an annotation.
    for notice in ratchet_notices(SUPERSEDED, counts, baseline):
        print(f"::notice::superseded-method ratchet: {notice}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
