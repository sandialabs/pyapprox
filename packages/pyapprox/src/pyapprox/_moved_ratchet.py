"""Keeping old paths of moved modules from spreading.

Two checks over the old prefixes in :data:`pyapprox._moved.MOVED`:

- **Ratchet.** :func:`count_references` counts references to each old
  prefix (imports, ``mock.patch`` strings, documentation) across the
  repository. The count is recorded in ``ratchet_baselines.txt`` as
  ``moved:<old prefix>=<count>``; under that file's rule, a commit that
  lowers a count lowers its baseline in the same commit, and no commit
  raises one. A prefix's alias retires when its count reaches zero.
- **Added lines.** ``python -m pyapprox._moved_ratchet --staged`` fails
  when a staged added line names an old prefix, so a session pays only
  for the paths it writes, never for migrating untouched files.

A line containing ``moved: allow`` is exempt from both, for text that
must name an old path (release notes, the old -> new table).
"""

import os
import re
import subprocess  # nosec B404: runs git with fixed arguments
import sys
from pathlib import Path
from typing import Mapping, Sequence

SUFFIXES = (".py", ".qmd", ".md", ".rst", ".toml", ".cfg", ".ini", ".yml", ".yaml")
SKIP_DIRS = {
    ".git", "__pycache__", "_site", "_freeze", ".quarto", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".claude-logs", "build", "dist",
    "node_modules",
}  # fmt: skip
# The table names every old prefix by construction.
EXEMPT = ("pyapprox/_moved.py",)
ALLOW = "moved: allow"
BASELINE_FILE = "ratchet_baselines.txt"
BASELINE_KEY = "moved:"


def _pattern(prefixes: Sequence[str]) -> re.Pattern[str]:
    """Matches an old prefix as a whole dotted name: not preceded by a
    name character or a dot, not followed by a name character. Longest
    prefixes first, so the most specific one is reported."""
    names = "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True))
    return re.compile(rf"(?<![\w.])({names})(?!\w)")


def _exempt(path: str) -> bool:
    return path.endswith(EXEMPT) or path.endswith(BASELINE_FILE)


def _files(root: Path) -> list[Path]:
    """Scanned files under ``root``; skipped directories are pruned, not
    walked and filtered."""
    files = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in SUFFIXES and not _exempt(path.as_posix()):
                files.append(path)
    return files


def count_references(table: Mapping[str, str], root: Path) -> dict[str, int]:
    """References to each old prefix in ``table`` under ``root``."""
    counts = dict.fromkeys(table, 0)
    if not table:
        return counts
    pattern = _pattern(list(table))
    for path in _files(root):
        for line in path.read_text(errors="ignore").splitlines():
            if ALLOW in line:
                continue
            for match in pattern.finditer(line):
                counts[match.group(1)] += 1
    return counts


def read_baseline(path: Path) -> dict[str, int]:
    """The ``moved:<prefix>=<count>`` entries of a baseline file."""
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
    baseline, below it (lower the baseline), or a prefix without one; and
    baselines of prefixes no longer moved."""
    problems = []
    for prefix in table:
        count, key = counts.get(prefix, 0), f"{BASELINE_KEY}{prefix}"
        if prefix not in baseline:
            problems.append(f"no baseline for {prefix}: add {key}={count}")
        elif count > baseline[prefix]:
            problems.append(
                f"{prefix}: {count} references, baseline {baseline[prefix]}; "
                "import the new path instead"
            )
        elif count < baseline[prefix]:
            problems.append(f"{prefix}: down to {count}; set {key}={count}")
    problems += [
        f"{BASELINE_KEY}{p} is stale: {p} is not moved"
        for p in baseline
        if p not in table
    ]
    return problems


def added_line_problems(table: Mapping[str, str], diff: str) -> list[str]:
    """``path:line: text`` for each added line of a unified diff naming
    an old prefix."""
    if not table:
        return []
    pattern = _pattern(list(table))
    problems, path, number = [], "", 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            hunk = re.search(r"\+(\d+)", line)
            number = int(hunk.group(1)) if hunk else 0
        elif line.startswith("+") and path:
            text = line[1:]
            if not _exempt(path) and ALLOW not in text and pattern.search(text):
                problems.append(f"{path}:{number}: {text.strip()}")
            number += 1
        elif line.startswith(" "):
            number += 1
    return problems


def main(argv: Sequence[str]) -> int:
    from pyapprox._moved import MOVED

    if "--staged" in argv:
        diff = subprocess.run(  # nosec B603 B607: fixed git arguments
            ["git", "diff", "--cached", "-U0", "--no-color"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        problems = added_line_problems(MOVED, diff)
        prefix = "old path of a moved module on an added line"
    else:
        root = Path(__file__).resolve().parents[4]
        counts = count_references(MOVED, root)
        problems = ratchet_problems(MOVED, counts, read_baseline(root / BASELINE_FILE))
        prefix = "moved-module ratchet"
    for problem in problems:
        print(f"{prefix}: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
