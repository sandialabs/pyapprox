"""Backend conformance: every row of the table in
``tests/_helpers/backend_conformance.py`` on every backend, and the
ratchets that keep every protocol method either covered by a row or
listed in the shrink-only ``NOT_YET_COVERED``, and every method
``JaxBkd`` lacks listed in the shrink-only ``NOT_YET_ON_JAX``."""

import ast
from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.backend_conformance import (
    CASES,
    NOT_YET_COVERED,
    NOT_YET_ON_JAX,
    ROWS_NOT_YET_ON_JAX,
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


def _jax_methods() -> frozenset[str]:
    """The public methods of ``JaxBkd``, read from its source, so the
    check runs without JAX installed."""
    source = Path(pyapprox.__file__).parent / "util" / "backends" / "jax.py"
    tree = ast.parse(source.read_text())
    (jax_bkd,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "JaxBkd"
    ]
    return frozenset(
        node.name
        for node in jax_bkd.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    )


def test_not_yet_on_jax_lists_exactly_what_jax_lacks() -> None:
    jax_methods = _jax_methods()
    methods = _protocol_methods()
    covered = {case.method for case in CASES}
    assert jax_methods <= methods, f"not protocol methods: {jax_methods - methods}"
    untested = jax_methods - covered
    assert not untested, f"JaxBkd methods without a row: {untested}"
    assert covered - jax_methods == ROWS_NOT_YET_ON_JAX, (
        "ROWS_NOT_YET_ON_JAX must list exactly the methods with a row that "
        f"JaxBkd lacks: {sorted(covered - jax_methods)}"
    )
    assert methods - jax_methods == NOT_YET_ON_JAX


@pytest.mark.parametrize(
    "key, listed",
    [
        ("conformance_uncovered", NOT_YET_COVERED),
        ("conformance_not_on_jax", NOT_YET_ON_JAX),
    ],
)
def test_lists_match_their_baselines(key: str, listed: frozenset[str]) -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    prefix = f"{key}="
    lines = (root / BASELINE_FILE).read_text().splitlines()
    entry = next(line for line in lines if line.startswith(prefix))
    baseline = int(entry[len(prefix) :])
    count = len(listed)
    assert count <= baseline, (
        f"{key} grew to {count} (baseline {baseline}); add rows and "
        "implementations instead"
    )
    assert count >= baseline, f"{key} shrank to {count}; set {prefix}{count}"
