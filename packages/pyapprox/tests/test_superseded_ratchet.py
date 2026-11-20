"""The superseded-method ratchet: calls to superseded backend methods
are counted in Python and Quarto sources and held at the baselines in
ratchet_baselines.txt."""

from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox._superseded_ratchet import (
    SUPERSEDED,
    count_calls,
    ratchet_problems,
    read_baseline,
)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_counts_calls_not_definitions(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/a.py",
        "def any_bool(x): ...\n"  # a definition
        "ok = bkd.any_bool(x) and bkd.any_bool(y)\n"
        "rows = self._bkd.all_array(x, 0)\n"
        "z = bkd.any(x)\n",  # the replacement
    )
    _write(tmp_path, "docs/page.qmd", "bkd.all_array(x, 1)\n")
    _write(tmp_path, "docs/notes.md", "bkd.any_bool(x)\n")  # not a call suffix
    _write(tmp_path, "build/gen.py", "bkd.any_bool(x)\n")  # skipped directory
    _write(tmp_path, "tests/test_superseded_ratchet.py", "bkd.any_bool(x)\n")
    counts = count_calls(["any_bool", "all_array", "all_bool"], tmp_path)
    assert counts == {"any_bool": 2, "all_array": 2, "all_bool": 0}
    assert count_calls([], tmp_path) == {}


def test_ratchet_holds_counts_at_their_baselines(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASELINE_FILE,
        "mypy_src=1\nmoved:pkg.old=4\nsuperseded:a=3\nsuperseded:gone=2\n",
    )
    baseline = read_baseline(tmp_path / BASELINE_FILE)
    assert baseline == {"a": 3, "gone": 2}
    table = {"a": "new_a(x)", "b": "new_b(x)"}
    problems = ratchet_problems(table, {"a": 4, "b": 1}, baseline)
    assert problems == [
        "a: 4 calls, baseline 3; use new_a(x) instead",
        "no baseline for b: add superseded:b=1",
        "superseded:gone is stale: gone is not superseded",
    ]
    fell = ratchet_problems({"a": "new_a(x)"}, {"a": 1}, {"a": 3})
    assert fell == ["a: down to 1; set superseded:a=1"]
    assert ratchet_problems({"a": "new_a(x)"}, {"a": 3}, {"a": 3}) == []


def test_repository_holds_its_baselines() -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    counts = count_calls(list(SUPERSEDED), root)
    assert (
        ratchet_problems(SUPERSEDED, counts, read_baseline(root / BASELINE_FILE)) == []
    )
