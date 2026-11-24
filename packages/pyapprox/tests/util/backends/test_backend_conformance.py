"""Backend conformance: every row of the table in
``tests/_helpers/backend_conformance.py`` on every backend, eagerly and
compiled through ``bkd.jit``, and the ratchets that keep every protocol
method either covered (by a row or a column) or listed in the
shrink-only ``NOT_YET_COVERED``, every method ``JaxBkd`` lacks listed
in the shrink-only ``NOT_YET_ON_JAX``, every ``ArrayProtocol`` member
covered by a member row or listed in ``NOT_YET_COVERED_ARRAY``, and
every public backend-class method on ``Backend`` or ``AutodiffBackend``
or listed in ``BACKEND_EXTRAS``."""

import ast
import functools
from pathlib import Path
from typing import Any, Callable, List

import pytest

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.backend_conformance import (
    ARRAY_PROPERTIES,
    BACKEND_EXTRAS,
    CASES,
    COVERED_BY_COLUMNS,
    NOT_YET_COVERED,
    NOT_YET_COVERED_ARRAY,
    NOT_YET_ON_JAX,
    ROWS_NOT_YET_ON_JAX,
    Case,
    ConformanceBackend,
    case_id,
    check_jacobian,
    jit_call,
    run_case,
)


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_conforms(bkd: Backend[Array], case: Case) -> None:
    run_case(bkd, case)


def _compiling_torch_bkd() -> ConformanceBackend[Any]:
    torch = pytest.importorskip("torch")
    from pyapprox.util.backends.torch import TorchBkd

    # One row per test: a fresh dynamo cache keeps the rows from adding
    # up to its recompile limit, past which it would run them eagerly.
    torch._dynamo.reset()
    return TorchBkd(compile=True)


def _jax_bkd() -> ConformanceBackend[Any]:
    pytest.importorskip("jax")
    from pyapprox.util.backends.jax import JaxBkd

    return JaxBkd()


@pytest.fixture(params=["NumpyBkd", "TorchBkd", "JaxBkd"])
def jit_bkd(request: pytest.FixtureRequest) -> ConformanceBackend[Any]:
    """Every backend with compilation on (NumPy has no compiler, so its
    jit runs the kernel as written); skipped where the library is
    missing."""
    if request.param == "TorchBkd":
        return _compiling_torch_bkd()
    if request.param == "JaxBkd":
        return _jax_bkd()
    return NumpyBkd()


_TRACEABLE = [case for case in CASES if not case.eager_only and case.raises is None]


_DIFFERENTIABLE = [case for case in CASES if case.differentiable]


@pytest.mark.parametrize("case", _DIFFERENTIABLE, ids=case_id)
def test_jacobian(autodiff_bkd: AutodiffBackend[Array], case: Case) -> None:
    """Every differentiable row's ``bkd.jacobian`` against central
    differences, on every autodiff backend."""
    check_jacobian(autodiff_bkd, case)


@pytest.mark.parametrize("case", _TRACEABLE, ids=case_id)
def test_conforms_compiled(jit_bkd: ConformanceBackend[Any], case: Case) -> None:
    """Every traceable row compiled through ``bkd.jit`` (torch at
    ``fullgraph=True``, so a graph break fails the row)."""
    if type(jit_bkd).__name__ == "JaxBkd" and case.method in NOT_YET_ON_JAX:
        pytest.skip("not on JaxBkd yet")
    run_case(jit_bkd, case, call=jit_call(jit_bkd, case))


def _scale(x: Any, *, bkd: Any) -> Any:
    return bkd.exp(x) * 2.0


def _reduce(x: Any, *, bkd: Any) -> Any:
    return bkd.sum(x, axis=0)


def _composite(x: Any, *, bkd: Any) -> Any:
    # Calls one kernel directly and one through jit, as code reached from
    # inside a kernel may: both are traced into the composite's graph.
    return _reduce(bkd.jit(_scale)(x), bkd=bkd)


class _Kernels:
    @staticmethod
    def scale(x: Any, *, bkd: Any) -> Any:
        return _scale(x, bkd=bkd)

    def bound(self, x: Any, *, bkd: Any) -> Any:
        return _scale(x, bkd=bkd)


class TestJit:
    """What ``bkd.jit`` does on every backend; backend-specific behavior
    is in the torch and JAX modules."""

    def test_binds_the_backend(self, jit_bkd: ConformanceBackend[Any]) -> None:
        x = jit_bkd.asarray([[0.0, 1.0], [2.0, -1.0]])
        jit_bkd.assert_allclose(jit_bkd.jit(_scale)(x), _scale(x, bkd=jit_bkd))

    def test_accepts_a_staticmethod(self, jit_bkd: ConformanceBackend[Any]) -> None:
        x = jit_bkd.asarray([0.5, 1.5])
        jit_bkd.assert_allclose(jit_bkd.jit(_Kernels.scale)(x), _scale(x, bkd=jit_bkd))

    def test_composite_compiles_as_one(self, jit_bkd: ConformanceBackend[Any]) -> None:
        # On torch fullgraph=True: a graph break at the nested jit raises.
        x = jit_bkd.asarray([[0.0, 1.0], [2.0, -1.0]])
        expected = _reduce(_scale(x, bkd=jit_bkd), bkd=jit_bkd)
        jit_bkd.assert_allclose(jit_bkd.jit(_composite)(x), expected)

    def test_refuses_what_is_new_per_call(
        self, jit_bkd: ConformanceBackend[Any]
    ) -> None:
        def local(x: Any, *, bkd: Any) -> Any:
            return x

        refused: List[Callable[..., Any]] = [
            lambda x, *, bkd: x,
            local,
            _Kernels().bound,
            functools.partial(_scale),
        ]
        for kernel in refused:
            with pytest.raises(TypeError, match="module-level function"):
                jit_bkd.jit(kernel)


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


def _covered() -> frozenset[str]:
    rows = frozenset(case.method for case in CASES if not case.member)
    return rows | COVERED_BY_COLUMNS


def test_every_protocol_method_has_a_row_or_is_listed() -> None:
    methods = _protocol_methods()
    covered = _covered()
    assert covered <= methods, f"rows for non-protocol methods: {covered - methods}"
    missing = methods - covered - NOT_YET_COVERED
    assert not missing, f"protocol methods with neither a row nor a listing: {missing}"
    stale = NOT_YET_COVERED & covered
    assert not stale, f"listed in NOT_YET_COVERED but have rows: {stale}"
    unknown = NOT_YET_COVERED - methods
    assert not unknown, f"listed in NOT_YET_COVERED but not protocol: {unknown}"


def _class_functions(module: str, name: str) -> List[ast.FunctionDef]:
    """The functions defined in the body of class ``name`` of the backend
    module ``module``, read from its source, so the checks run without
    the module's library installed."""
    source = Path(pyapprox.__file__).parent / "util" / "backends" / module
    tree = ast.parse(source.read_text())
    (cls,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    return [node for node in cls.body if isinstance(node, ast.FunctionDef)]


def _public_methods(module: str, name: str) -> frozenset[str]:
    return frozenset(
        node.name
        for node in _class_functions(module, name)
        if not node.name.startswith("_")
    )


def test_not_yet_on_jax_lists_exactly_what_jax_lacks() -> None:
    jax_methods = _public_methods("jax.py", "JaxBkd")
    methods = _protocol_methods()
    covered = _covered()
    assert jax_methods <= methods, f"not protocol methods: {jax_methods - methods}"
    untested = jax_methods - covered
    assert not untested, f"JaxBkd methods without a row: {untested}"
    assert covered - jax_methods == ROWS_NOT_YET_ON_JAX, (
        "ROWS_NOT_YET_ON_JAX must list exactly the methods with a row that "
        f"JaxBkd lacks: {sorted(covered - jax_methods)}"
    )
    assert methods - jax_methods == NOT_YET_ON_JAX


def _array_members() -> List[ast.FunctionDef]:
    return _class_functions("protocols.py", "ArrayProtocol")


def test_every_array_member_has_a_row_or_is_listed() -> None:
    members = frozenset(node.name for node in _array_members())
    rows = frozenset(case.method for case in CASES if case.member)
    assert rows <= members, f"member rows for non-members: {rows - members}"
    missing = members - rows - NOT_YET_COVERED_ARRAY
    assert not missing, f"members with neither a row nor a listing: {missing}"
    stale = NOT_YET_COVERED_ARRAY & rows
    assert not stale, f"listed in NOT_YET_COVERED_ARRAY but have rows: {stale}"
    unknown = NOT_YET_COVERED_ARRAY - members
    assert not unknown, f"listed in NOT_YET_COVERED_ARRAY but not members: {unknown}"


def test_array_properties_are_the_protocol_properties() -> None:
    properties = frozenset(
        node.name
        for node in _array_members()
        if any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        )
    )
    assert ARRAY_PROPERTIES == properties


@pytest.mark.parametrize(
    "module, name",
    [("numpy.py", "NumpyBkd"), ("torch.py", "TorchBkd"), ("jax.py", "JaxBkd")],
)
def test_backend_methods_are_on_a_protocol_or_listed(module: str, name: str) -> None:
    protocols = _protocol_methods() | frozenset(
        n for n in vars(AutodiffBackend) if not n.startswith("_")
    )
    extras = _public_methods(module, name) - protocols
    assert extras == BACKEND_EXTRAS[name], (
        f"BACKEND_EXTRAS[{name!r}] must list exactly the public methods on "
        f"no protocol: {sorted(extras)}"
    )


@pytest.mark.parametrize(
    "key, listed",
    [
        ("conformance_uncovered", NOT_YET_COVERED),
        ("conformance_not_on_jax", NOT_YET_ON_JAX),
        ("conformance_array_uncovered", NOT_YET_COVERED_ARRAY),
        (
            "backend_extras",
            frozenset(
                f"{name}.{method}"
                for name, methods in BACKEND_EXTRAS.items()
                for method in methods
            ),
        ),
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
