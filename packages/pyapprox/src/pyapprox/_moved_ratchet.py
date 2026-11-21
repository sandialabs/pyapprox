"""Keeping old paths of moved modules, and superseded packages, from
spreading.

Two tables of prefixes:

- :data:`pyapprox._moved.MOVED`, old paths of moved modules, which still
  import through a runtime alias; baseline key ``moved:``.
- :data:`SUPERSEDED`, packages replaced by another but not moved: they
  have no alias and no new code may use them; baseline key
  ``superseded:``. References inside a superseded package itself (its
  own modules, and those of the other superseded packages) are not
  counted, since they go when the package does.

Two checks over each:

- **Ratchet.** :func:`count_references` counts references to each
  prefix (imports, ``mock.patch`` strings, documentation) across the
  repository. The count is recorded in ``ratchet_baselines.txt`` as
  ``<key><prefix>=<count>``; under that file's rule, a commit that
  lowers a count lowers its baseline in the same commit, and no commit
  raises one. A moved prefix's alias retires, and a superseded package
  can be deleted, when its count reaches zero.
- **Added lines.** ``python -m pyapprox._moved_ratchet --staged`` fails
  when a staged added line names a prefix, so a session pays only for
  the paths it writes, never for migrating untouched files.

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
# .claude holds per-checkout agent configuration that is never tracked,
# so counting it would make a local count exceed CI's.
SKIP_DIRS = {
    ".git", "__pycache__", "_site", "_freeze", ".quarto", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".claude", ".claude-logs", "build",
    "dist", "node_modules",
}  # fmt: skip
# The tables name every prefix by construction.
EXEMPT = ("pyapprox/_moved.py", "pyapprox/_moved_ratchet.py")
ALLOW = "moved: allow"
BASELINE_FILE = "ratchet_baselines.txt"
BASELINE_KEY = "moved:"
SUPERSEDED_KEY = "superseded:"
# Superseded package -> what replaces it.
SUPERSEDED = {
    "pyapprox.pde.manufactured": "pyapprox.pde.verification",
    "pyapprox.pde.galerkin.manufactured": "pyapprox.pde.verification",
}


def _inside(path: str, packages: Sequence[str]) -> bool:
    """Whether ``path`` is a file of one of the dotted ``packages``."""
    slashed = "/" + path
    return any(f"/{p.replace('.', '/')}/" in slashed for p in packages)


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


def count_references(
    table: Mapping[str, str], root: Path, skip_inside: bool = False
) -> dict[str, int]:
    """References to each prefix in ``table`` under ``root``; with
    ``skip_inside``, not counting files of the table's own packages."""
    counts = dict.fromkeys(table, 0)
    if not table:
        return counts
    pattern = _pattern(list(table))
    for path in _files(root):
        if skip_inside and _inside(path.as_posix(), list(table)):
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if ALLOW in line:
                continue
            for match in pattern.finditer(line):
                counts[match.group(1)] += 1
    return counts


def read_baseline(path: Path, key: str = BASELINE_KEY) -> dict[str, int]:
    """The ``<key><prefix>=<count>`` entries of a baseline file."""
    baseline: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith(key) and "=" in line:
            prefix, value = line[len(key) :].split("=", 1)
            baseline[prefix.strip()] = int(value)
    return baseline


def ratchet_problems(
    table: Mapping[str, str],
    counts: Mapping[str, int],
    baseline: Mapping[str, int],
    key: str = BASELINE_KEY,
) -> list[str]:
    """What differs between ``counts`` and ``baseline``: a count above its
    baseline, below it (lower the baseline), or a prefix without one; and
    baselines of prefixes no longer in ``table``."""
    moved = key == BASELINE_KEY
    advice = "import the new path instead" if moved else "use {} instead"
    status = "moved" if moved else "superseded"
    problems = []
    for prefix in table:
        count, entry = counts.get(prefix, 0), f"{key}{prefix}"
        if prefix not in baseline:
            problems.append(f"no baseline for {prefix}: add {entry}={count}")
        elif count > baseline[prefix]:
            problems.append(
                f"{prefix}: {count} references, baseline {baseline[prefix]}; "
                + advice.format(table[prefix])
            )
        elif count < baseline[prefix]:
            problems.append(f"{prefix}: down to {count}; set {entry}={count}")
    problems += [
        f"{key}{p} is stale: {p} is not {status}" for p in baseline if p not in table
    ]
    return problems


def added_line_problems(
    table: Mapping[str, str], diff: str, skip_inside: bool = False
) -> list[str]:
    """``path:line: text`` for each added line of a unified diff naming
    a prefix of ``table``; with ``skip_inside``, not in files of the
    table's own packages."""
    if not table:
        return []
    pattern = _pattern(list(table))
    problems, path, number = [], "", 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
            if skip_inside and _inside(path, list(table)):
                path = ""
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
        reports = [
            (
                "old path of a moved module on an added line",
                added_line_problems(MOVED, diff),
            ),
            (
                "superseded package on an added line",
                added_line_problems(SUPERSEDED, diff, skip_inside=True),
            ),
        ]
    else:
        root = Path(__file__).resolve().parents[4]
        baselines = root / BASELINE_FILE
        moved = count_references(MOVED, root)
        superseded = count_references(SUPERSEDED, root, skip_inside=True)
        reports = [
            (
                "moved-module ratchet",
                ratchet_problems(MOVED, moved, read_baseline(baselines)),
            ),
            (
                "superseded-package ratchet",
                ratchet_problems(
                    SUPERSEDED,
                    superseded,
                    read_baseline(baselines, SUPERSEDED_KEY),
                    SUPERSEDED_KEY,
                ),
            ),
        ]
    failed = False
    for prefix, problems in reports:
        for problem in problems:
            print(f"{prefix}: {problem}")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
