"""Backend conformance: every row of the table in
``tests/_helpers/backend_conformance.py`` on every backend, and the
ratchet that keeps every protocol method either covered by a row or
listed in its shrink-only ``NOT_YET_COVERED``."""

from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.backend_conformance import (
    CASES,
    NOT_YET_COVERED,
    Case,
    case_id,
    run_case,
)


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_conforms(bkd: Backend[Array], case: Case) -> None:
    run_case(bkd, case)


def _protocol_methods() -> frozenset[str]:
    return frozenset(n for n in vars(Backend) if not n.startswith("_"))


def test_case_ids_are_unique() -> None:
    ids = [case_id(case) for case in CASES]
    assert len(ids) == len(set(ids))


def test_assert_allclose_rows_come_first() -> None:
    # Every later row is checked with assert_allclose, so its own rows,
    # checked with pytest.raises, run before any of them.
    methods = [case.method for case in CASES]
    count = methods.count("assert_allclose")
    assert methods[:count] == ["assert_allclose"] * count


def test_every_protocol_method_has_a_row_or_is_listed() -> None:
    methods = _protocol_methods()
    covered = {case.method for case in CASES}
    assert covered <= methods, f"rows for non-protocol methods: {covered - methods}"
    missing = methods - covered - NOT_YET_COVERED
    assert not missing, f"protocol methods with neither a row nor a listing: {missing}"
    stale = NOT_YET_COVERED & covered
    assert not stale, f"listed in NOT_YET_COVERED but have rows: {stale}"
    unknown = NOT_YET_COVERED - methods
    assert not unknown, f"listed in NOT_YET_COVERED but not protocol: {unknown}"


def test_not_yet_covered_matches_its_baseline() -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    key = "conformance_uncovered="
    lines = (root / BASELINE_FILE).read_text().splitlines()
    baseline = int(next(line for line in lines if line.startswith(key))[len(key) :])
    count = len(NOT_YET_COVERED)
    assert count <= baseline, (
        f"NOT_YET_COVERED grew to {count} (baseline {baseline}); add rows instead"
    )
    assert count >= baseline, (
        f"NOT_YET_COVERED shrank to {count}; set {key}{count} in {BASELINE_FILE}"
    )
