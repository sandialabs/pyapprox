"""Keeping calls to superseded backend methods from spreading.

:data:`SUPERSEDED` names the ``Backend`` methods kept only until their
callers are converted, each with its replacement. :func:`count_calls`
counts calls to each (``.name(``, so definitions do not count) in the
repository's Python and Quarto sources. The counts are recorded in
``ratchet_baselines.txt`` as ``superseded:<name>=<count>``; under that
file's rule, a commit that lowers a count lowers its baseline in the
same commit, and no commit raises one. A method is deleted from the
backends when its count reaches zero.

``python -m pyapprox._superseded_ratchet`` exits nonzero when a count
differs from its baseline.
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
BASELINE_KEY = "superseded:"
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
    """The ``superseded:<name>=<count>`` entries of a baseline file."""
    baseline: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith(BASELINE_KEY) and "=" in line:
            key, value = line[len(BASELINE_KEY) :].split("=", 1)
            baseline[key.strip()] = int(value)
    return baseline


def ratchet_problems(
    table: Mapping[str, str], counts: Mapping[str, int], baseline: Mapping[str, int]
) -> list[str]:
    """What differs between ``counts`` and ``baseline``: a count above its
    baseline, below it (lower the baseline), or a method without one;
    and baselines of methods no longer superseded."""
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
        elif count < baseline[name]:
            problems.append(f"{name}: down to {count}; set {key}={count}")
    problems += [
        f"{BASELINE_KEY}{n} is stale: {n} is not superseded"
        for n in baseline
        if n not in table
    ]
    return problems


def main(argv: Sequence[str]) -> int:
    root = Path(__file__).resolve().parents[4]
    counts = count_calls(list(SUPERSEDED), root)
    problems = ratchet_problems(SUPERSEDED, counts, read_baseline(root / BASELINE_FILE))
    for problem in problems:
        print(f"superseded-method ratchet: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
