"""Backend conformance: every protocol method against a NumPy reference.

:data:`CASES` is one table of rows. A row names a ``Backend`` method, its
inputs as NumPy arrays and Python values, and the reference: by default
the ``numpy`` function of the same name applied to the same inputs in
float64. One parametrized test runs each row on every backend and checks,
with the backend's own strict ``assert_allclose``,

- the values, within ``10 * eps`` of the reference dtype for floating
  results and exactly otherwise, the exact shape, and the result dtype's
  role (:class:`Expect`): the reference is converted to that dtype on the
  backend, and ``assert_allclose`` refuses a shape or dtype mismatch;
- that the array inputs are unchanged;
- the exception class, for rows that expect one.

So ``asarray`` and ``assert_allclose`` are the first methods a backend
implements. The harness trusts ``assert_allclose`` only after its own
rows, which come first and are checked with ``pytest.raises``: it must
refuse a value outside the tolerance, a shape mismatch and a dtype
mismatch, so a lenient implementation fails there before it can pass
the rest of the table. The dtype accessors are the one host check: a
dtype object is compared through the NumPy dtype of an array created
with it.

Where the backends diverge, NumPy's semantics are the contract and the
other backends are fixed to match.

:data:`NOT_YET_COVERED` lists the protocol methods without a row. It may
only shrink: its length is the ``conformance_uncovered`` baseline in
``ratchet_baselines.txt``, and a new protocol method cannot enter it, so
it arrives with a row.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pytest
from numpy.typing import NDArray

import pyapprox
from pyapprox._moved_ratchet import BASELINE_FILE
from pyapprox.util.backends.protocols import Array, Backend


class Role(Enum):
    """A dtype named by its role, resolved on each backend."""

    DEFAULT = "default"
    COMPLEX = "complex"
    INDEX = "index"
    BOOL = "bool"


class Expect(Enum):
    """What a row's result is: an array whose dtype has a role, or
    something else."""

    # The dtype of the first non-boolean array input.
    INPUT = "input"
    DEFAULT = "default"
    COMPLEX = "complex"
    INDEX = "index"
    BOOL = "bool"
    # A dtype object (the dtype accessors).
    DTYPE = "dtype"
    # A Python value or NumPy array, compared as it is.
    HOST = "host"
    # Nothing (``assert_allclose``).
    NONE = "none"


Arg = Union[
    NDArray[Any], Role, str, int, float, slice, None, Tuple["Arg", ...], List[Any]
]


@dataclass(frozen=True)
class Case:
    """One conformance row.

    ``reference`` receives ``args`` and ``kwargs`` with roles resolved to
    NumPy dtypes; ``None`` means the ``numpy`` function named ``method``.
    ``eager_only`` marks rows that cannot be traced (data-dependent
    shape, host conversion); ``differentiable`` rows have floating inputs
    a gradient can flow through.
    """

    method: str
    id: str
    args: Tuple[Arg, ...] = ()
    kwargs: Mapping[str, Arg] = field(default_factory=dict)
    reference: Optional[Callable[..., Any]] = None
    expect: Expect = Expect.INPUT
    raises: Optional[type[Exception]] = None
    eager_only: bool = False
    differentiable: bool = False


def _rand(shape: Tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(shape)


def _crand(shape: Tuple[int, ...], seed: int = 0) -> NDArray[np.complex128]:
    return _rand(shape, seed) + 1j * _rand(shape, seed + 1)


def _index(*values: int) -> NDArray[np.int64]:
    return np.asarray(values, dtype=np.int64)


def _index_add(
    array: NDArray[Any], index: NDArray[np.int64], values: NDArray[Any]
) -> NDArray[Any]:
    result = array.copy()
    np.add.at(result, index, values)
    return result


def _index_update(array: NDArray[Any], index: Any, value: Any) -> NDArray[Any]:
    result = array.copy()
    result[index] = value
    return result


def _where(
    condition: NDArray[np.bool_],
    x: Optional[Union[NDArray[Any], float]] = None,
    y: Optional[Union[NDArray[Any], float]] = None,
) -> Any:
    if x is None:
        return np.nonzero(condition)
    if y is None:
        raise ValueError("Must provide both x and y or neither.")
    return np.where(condition, x, y)


def _none(*args: Any, **kwargs: Any) -> None:
    return None


_A = _rand((2, 3))
_B = _rand((3, 4), seed=1)
_POS = np.abs(_rand((2, 3), seed=2)) + 0.5
_T3 = _rand((2, 3, 4), seed=3)
_C = _crand((2, 3), seed=4)
_MASK = np.asarray([[True, False, True], [False, False, True]])
_NAN = np.asarray([1.0, np.nan, 3.0])


def _reductions(method: str) -> List[Case]:
    return [
        Case(method, "axis-none", (_A,), differentiable=True),
        Case(method, "axis-0", (_A,), {"axis": 0}, differentiable=True),
        Case(method, "axis-neg", (_T3,), {"axis": -1}, differentiable=True),
        Case(method, "keepdims", (_A,), {"axis": 1, "keepdims": True}),
        Case(method, "complex", (_C,), {"axis": 0}),
    ]


CASES: List[Case] = [
    # ---- assert_allclose first: the harness checks every later row with it
    Case("assert_allclose", "equal", (_A, _A.copy()), reference=_none,
         expect=Expect.NONE, eager_only=True),
    Case("assert_allclose", "within-rtol", (_A, _A * (1.0 + 1e-9)),
         {"rtol": 1e-8}, reference=_none, expect=Expect.NONE, eager_only=True),
    Case("assert_allclose", "nan-equal", (_NAN, _NAN.copy()), reference=_none,
         expect=Expect.NONE, eager_only=True),
    Case("assert_allclose", "index-equal", (_index(1, 2), _index(1, 2)),
         reference=_none, expect=Expect.NONE, eager_only=True),
    Case("assert_allclose", "value", (_A, _A + 1e-3), raises=AssertionError,
         eager_only=True),
    Case("assert_allclose", "beyond-rtol", (_A, _A * (1.0 + 1e-6)),
         {"rtol": 1e-8}, raises=AssertionError, eager_only=True),
    Case("assert_allclose", "shape-0d-vs-2d", (np.asarray(1.0), np.ones((1, 1))),
         raises=AssertionError, eager_only=True),
    Case("assert_allclose", "shape-transposed", (_A, _A.reshape(3, 2)),
         raises=AssertionError, eager_only=True),
    Case("assert_allclose", "dtype", (np.ones(2), np.ones(2, dtype=np.int64)),
         raises=AssertionError, eager_only=True),
    Case("assert_allclose", "nan-vs-value", (_NAN, np.asarray([1.0, 2.0, 3.0])),
         raises=AssertionError, eager_only=True),
    # ---- dtypes and conversion
    Case("default_dtype", "f64", reference=lambda: np.float64, expect=Expect.DTYPE),
    Case("double_dtype", "f64", reference=lambda: np.float64, expect=Expect.DTYPE),
    Case("complex_dtype", "c128", reference=lambda: np.complex128,
         expect=Expect.DTYPE),
    Case("int64_dtype", "i64", reference=lambda: np.int64, expect=Expect.DTYPE),
    Case("asarray", "float-list", ([[1.0, 2.0], [3.0, 4.0]],),
         reference=lambda x: np.asarray(x, dtype=np.float64),
         expect=Expect.DEFAULT, eager_only=True),
    Case("asarray", "index", (_index(3, 1, 2),), {"dtype": Role.INDEX},
         expect=Expect.INDEX),
    Case("asarray", "complex", (_C,), {"dtype": Role.COMPLEX}),
    Case("to_numpy", "float", (_A,), reference=lambda x: x, expect=Expect.HOST,
         eager_only=True),
    Case("to_bool", "true", (np.asarray([[2.0]]),), reference=lambda x: True,
         expect=Expect.HOST, eager_only=True),
    Case("to_bool", "false-0d", (np.asarray(0.0),), reference=lambda x: False,
         expect=Expect.HOST, eager_only=True),
    Case("to_bool", "several", (_A,), raises=ValueError, eager_only=True),
    Case("to_bool", "empty", (np.zeros((0,)),), raises=ValueError,
         eager_only=True),
    # ---- creation
    Case("zeros", "default", ((2, 3),), expect=Expect.DEFAULT),
    Case("zeros", "index", ((4,),), {"dtype": Role.INDEX}, expect=Expect.INDEX),
    Case("ones", "default", ((3, 1),), expect=Expect.DEFAULT),
    Case("ones", "complex", ((2,),), {"dtype": Role.COMPLEX},
         expect=Expect.COMPLEX),
    Case("full", "default", ((2, 2), 3.5), expect=Expect.DEFAULT),
    Case("full", "index", ((3,), 7), {"dtype": Role.INDEX}, expect=Expect.INDEX),
    # ---- shape
    Case("reshape", "2d", (_A, (3, 2)), differentiable=True),
    Case("reshape", "inferred", (_T3, (-1, 4)), differentiable=True),
    Case("reshape", "index", (_index(1, 2, 3, 4), (2, 2))),
    Case("stack", "axis-0", ((_A, _POS),), differentiable=True),
    Case("stack", "axis-1", ((_A, _POS),), {"axis": 1}, differentiable=True),
    Case("concatenate", "axis-0", ((_A, _POS),), differentiable=True),
    Case("concatenate", "axis-1", ((_A, _B[:2]),), {"axis": 1},
         differentiable=True),
    Case("moveaxis", "int", (_T3, 0, 2), differentiable=True),
    Case("moveaxis", "tuple", (_T3, (0, 1), (2, 0)), differentiable=True),
    # ---- elementwise and contraction
    Case("sqrt", "float", (_POS,), differentiable=True),
    Case("sqrt", "complex", (_C,)),
    Case("exp", "float", (_A,), differentiable=True),
    Case("exp", "complex", (_C,)),
    Case("dot", "matrix-matrix", (_A, _B), differentiable=True),
    Case("dot", "matrix-vector", (_A, _B[:, 0]), differentiable=True),
    Case("einsum", "matmul", ("ij,jk->ik", _A, _B), differentiable=True),
    Case("einsum", "batched", ("nij,nkj->nik", _T3, _T3), differentiable=True),
    Case("einsum", "trace", ("ii->", _B[:, :3]), differentiable=True),
    Case("einsum", "outer", ("i,j->ij", _A[0], _B[0]), differentiable=True),
    Case("einsum", "complex", ("ij,kj->ik", _C, _C)),
    Case("where", "three-arrays", (_MASK, _A, _POS), reference=_where,
         differentiable=True),
    Case("where", "scalar-y", (_MASK, _A, 0.0), reference=_where),
    Case("where", "nonzero", (_MASK,), reference=_where, expect=Expect.INDEX,
         eager_only=True),
    Case("where", "x-without-y", (_MASK, _A), raises=ValueError),
    # ---- reductions
    *_reductions("sum"),
    Case("sum", "axis-tuple", (_T3,), {"axis": (0, 2)}, differentiable=True),
    *_reductions("mean"),
    Case("mean", "axis-tuple", (_T3,), {"axis": (0, 2)}, differentiable=True),
    Case("prod", "axis-none", (_POS,), differentiable=True),
    Case("prod", "axis-1", (_POS,), {"axis": 1}, differentiable=True),
    Case("prod", "keepdims", (_POS,), {"axis": 0, "keepdims": True}),
    Case("any", "axis-none", (_A > 0.5,), expect=Expect.BOOL),
    Case("any", "axis-tuple", (_T3 > 0.5,), {"axis": (1, 2)},
         expect=Expect.BOOL),
    Case("any", "float-input", (_A,), {"axis": 0}, expect=Expect.BOOL),
    Case("all", "keepdims", (_MASK,), {"axis": 1, "keepdims": True},
         expect=Expect.BOOL),
    Case("all", "empty", (np.zeros((0, 3)),), expect=Expect.BOOL),
    # ---- functional updates
    Case("index_update", "int", (_A[0], 1, 9.0), reference=_index_update,
         differentiable=True),
    Case("index_update", "slice", (_A, slice(0, 1), _POS[1]),
         reference=_index_update, differentiable=True),
    Case("index_add", "repeated", (_A[0], _index(2, 0, 2), _POS[0]),
         reference=_index_add, differentiable=True),
    Case("index_add", "rows", (_B, _index(1, 1, 0), _rand((3, 4), seed=5)),
         reference=_index_add,
         differentiable=True),
    Case("index_add", "complex", (_C[0], _index(0, 0, 1), _C[1]),
         reference=_index_add),
    Case("index_add", "2d-index",
         (_A[0], np.asarray([[0, 1]], dtype=np.int64), _POS[:1, :2]),
         raises=ValueError),
    Case("index_add", "row-mismatch", (_A[0], _index(0, 1), _POS[0]),
         raises=ValueError),
]  # fmt: skip

# Protocol methods without a row. Shrink-only: see the module docstring.
NOT_YET_COVERED = frozenset(
    {
        "abs", "all_array", "all_bool", "allclose", "any_array", "any_bool",
        "arange", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2",
        "arctanh", "argmax", "argmin", "argsort", "array", "array_type",
        "atleast_1d", "atleast_2d", "block", "cdist", "ceil", "cholesky",
        "cholesky_solve", "clip", "cond", "conj", "copy", "cos", "cosh", "cov",
        "cumsum", "det", "diag", "diff", "digamma", "eigh", "eigvalsh", "empty",
        "equal", "erf",
        "erfinv", "expm1", "eye", "flatten", "flip", "floor", "floor_divide",
        "full_like", "gammaln", "get_diagonal", "hstack", "inv",
        "is_floating_dtype", "is_integer_dtype", "isfinite", "isinf", "isnan",
        "kron", "linspace", "log", "log1p", "log_ndtr", "logspace", "logsumexp",
        "lstsq", "lu", "lu_factor", "lu_solve", "max", "maximum", "meshgrid",
        "min", "minimum", "multidot", "ndtr", "nonzero", "norm", "ones_like",
        "outer", "pinv", "qr", "quantile", "rank", "ravel", "repeat", "round",
        "searchsorted", "sign", "sin", "sinh", "slogdet", "solve", "solve_sparse",
        "solve_triangular", "sort", "split", "squeeze", "std", "svd", "tanh",
        "tile", "to_float", "to_int", "trace", "tracks_gradient", "transpose",
        "tril", "tril_indices", "triu", "unique", "var", "vstack", "zeros_like",
    }
)  # fmt: skip


# ---- running a row


def _bool_dtype(bkd: Backend[Array]) -> Any:
    # A comparison is the backend-neutral way to name the boolean dtype.
    return (bkd.zeros((1,)) > 0).dtype


def _backend_dtype(bkd: Backend[Array], role: Role) -> Any:
    if role is Role.DEFAULT:
        return bkd.default_dtype()
    if role is Role.COMPLEX:
        return bkd.complex_dtype()
    if role is Role.INDEX:
        return bkd.int64_dtype()
    return _bool_dtype(bkd)


_NUMPY_DTYPES = {
    Role.DEFAULT: np.float64,
    Role.COMPLEX: np.complex128,
    Role.INDEX: np.int64,
    Role.BOOL: np.bool_,
}
_ROLE_OF_NUMPY = {np.dtype(v): k for k, v in _NUMPY_DTYPES.items()}


def _to_backend(bkd: Backend[Array], value: Any) -> Any:
    """``value`` with NumPy arrays (also inside tuples) on ``bkd`` in the
    matching role's dtype, and roles resolved to ``bkd`` dtypes."""
    if isinstance(value, Role):
        return _backend_dtype(bkd, value)
    if isinstance(value, np.ndarray):
        role = _ROLE_OF_NUMPY[value.dtype]
        return bkd.asarray(value, dtype=_backend_dtype(bkd, role))
    if isinstance(value, tuple) and any(isinstance(v, np.ndarray) for v in value):
        return tuple(_to_backend(bkd, v) for v in value)
    return value


def _to_reference(value: Any) -> Any:
    return _NUMPY_DTYPES[value] if isinstance(value, Role) else value


def _reference(case: Case) -> Any:
    function = case.reference or getattr(np, case.method)
    args = [_to_reference(a) for a in case.args]
    kwargs = {k: _to_reference(v) for k, v in case.kwargs.items()}
    return function(*args, **kwargs)


def _arrays(values: Tuple[Any, ...]) -> List[Any]:
    """The backend arrays among ``values``, also inside tuples."""
    found: List[Any] = []
    for value in values:
        if isinstance(value, tuple):
            found += _arrays(value)
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            found.append(value)
    return found


def _expected_dtype(bkd: Backend[Array], case: Case, args: Tuple[Any, ...]) -> Any:
    if case.expect is Expect.INPUT:
        bool_dtype = _bool_dtype(bkd)
        return next(a.dtype for a in _arrays(args) if a.dtype != bool_dtype)
    return _backend_dtype(bkd, Role(case.expect.value))


def _assert_matches(
    bkd: Backend[Array], actual: Any, expected: Any, dtype: Any
) -> None:
    """``actual`` equals ``expected`` (NumPy, converted to ``dtype`` on
    ``bkd``) in shape, dtype and values; tuples elementwise."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_matches(bkd, a, e, dtype)
        return
    expected = np.asarray(expected)
    tol = 0.0
    if np.issubdtype(expected.dtype, np.inexact):
        tol = 10 * float(np.finfo(expected.dtype).eps)
    desired = bkd.asarray(expected, dtype=dtype)
    bkd.assert_allclose(actual, desired, rtol=tol, atol=tol)


def _assert_host_matches(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected), (type(actual), type(expected))
    np.testing.assert_array_equal(actual, expected, strict=True)


def _assert_dtype_object(bkd: Backend[Array], actual: Any, expected: Any) -> None:
    created = bkd.to_numpy(bkd.zeros((1,), dtype=actual)).dtype
    assert created == np.dtype(expected), (created, expected)


def run_case(
    bkd: Backend[Array],
    case: Case,
    call: Optional[Callable[[Callable[..., Any], Tuple[Any, ...]], Any]] = None,
) -> Any:
    """Run ``case`` on ``bkd`` and check it; returns the result.

    ``call(method, args)`` invokes the bound method (keyword arguments
    already bound) on the converted positional arguments, so a backend's
    own test can run it compiled.
    """
    args = tuple(_to_backend(bkd, a) for a in case.args)
    kwargs = {k: _to_backend(bkd, v) for k, v in case.kwargs.items()}
    bound: Callable[..., Any] = getattr(bkd, case.method)

    def method(*positional: Any) -> Any:
        return bound(*positional, **kwargs)

    invoke = call or (lambda fn, values: fn(*values))
    if case.raises is not None:
        with pytest.raises(case.raises):
            invoke(method, args)
        return None
    result = invoke(method, args)
    expected = _reference(case)
    if case.expect is Expect.NONE:
        assert result is None
    elif case.expect is Expect.HOST:
        _assert_host_matches(result, expected)
    elif case.expect is Expect.DTYPE:
        _assert_dtype_object(bkd, result, expected)
    else:
        _assert_matches(bkd, result, expected, _expected_dtype(bkd, case, args))
    # The inputs are unchanged: compare each with a fresh conversion.
    fresh = _arrays(tuple(_to_backend(bkd, a) for a in case.args))
    for array, original in zip(_arrays(args), fresh):
        bkd.assert_allclose(array, original, rtol=0.0, atol=0.0)
    return result


def _case_id(case: Case) -> str:
    return f"{case.method}-{case.id}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_conforms(bkd: Backend[Array], case: Case) -> None:
    run_case(bkd, case)


# ---- the coverage ratchet


def _protocol_methods() -> frozenset[str]:
    return frozenset(n for n in vars(Backend) if not n.startswith("_"))


def test_case_ids_are_unique() -> None:
    ids = [_case_id(case) for case in CASES]
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
