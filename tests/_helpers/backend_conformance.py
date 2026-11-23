"""Backend conformance table: every protocol method against a NumPy
reference.

:data:`CASES` is one table of rows. A row names a ``Backend`` method, its
inputs as NumPy arrays and Python values, and the reference: by default
the ``numpy`` function of the same name applied to the same inputs in
float64. :func:`run_case` runs a row on a backend and checks, with the
backend's own strict ``assert_allclose``,

- the values, within ``10 * eps`` of the reference dtype for floating
  results and exactly otherwise, the exact shape, and the result dtype's
  role (:class:`Expect`): the reference is converted to that dtype on the
  backend, and ``assert_allclose`` refuses a shape or dtype mismatch;
- that the array inputs are unchanged;
- the exception class, for rows that expect one.

A factorization unique only up to the signs of its vectors (``eigh``,
``svd``, ``qr``) is compared after its row's ``canonical`` fixes those
signs on the result and the reference alike. A method the protocol lets
a backend decline (``solve_sparse``) has ``optional`` rows: a backend
raising ``NotImplementedError`` there skips them.

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

The tests are in ``packages/pyapprox/tests/util/backends/``:
``test_backend_conformance.py`` runs every row on every backend, every
traceable row compiled through ``bkd.jit`` (:func:`row_kernel`), every
differentiable row's ``bkd.jacobian`` against central differences
(:func:`check_jacobian`), and the ratchets;
``test_backend_conformance_jax.py`` runs the rows on ``JaxBkd`` until
it joins the ``bkd`` fixture.

This module depends only on the backend package, NumPy, SciPy (which
the backend package requires) and pytest, so the backend package could
ship it as a public suite for other backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pytest
import scipy.linalg
import scipy.special
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.errors import LinAlgError
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, ArrayLike, KernelResult


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
    # A class every array the backend creates is an instance of
    # (``array_type``).
    ARRAY_TYPE = "array_type"
    # Nothing (``assert_allclose``).
    NONE = "none"


@dataclass(frozen=True, eq=False)
class Sparse:
    """A dense NumPy matrix that a row passes as a SciPy CSC matrix, to
    the method and to the reference alike."""

    dense: NDArray[np.float64]

    def to_scipy(self) -> csc_matrix:
        return csc_matrix(self.dense)


Arg = Union[
    NDArray[Any],
    Role,
    Sparse,
    str,
    int,
    float,
    slice,
    None,
    Tuple["Arg", ...],
    List[Any],
]


@dataclass(frozen=True)
class Case:
    """One conformance row.

    ``reference`` receives ``args`` and ``kwargs`` with roles resolved to
    NumPy dtypes; ``None`` means the ``numpy`` function named ``method``.
    ``eager_only`` marks rows that cannot be traced (data-dependent
    shape, host conversion); ``differentiable`` rows have floating inputs
    a gradient can flow through. ``parts`` gives the dtype role of each
    element of a tuple result whose elements differ (factors and
    pivots); otherwise every element has the role ``expect``.
    ``canonical(bkd, result)`` removes a sign ambiguity before the
    comparison; it is applied to the reference with ``NumpyBkd``.
    ``optional`` rows test a method a backend may decline by raising
    ``NotImplementedError``.
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
    parts: Tuple[Expect, ...] = ()
    canonical: Optional[Callable[[Any, Any], Any]] = None
    optional: bool = False


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
# Linear algebra: well conditioned, symmetric positive definite, its
# Cholesky factor, singular, and right-hand sides.
_GEN = _rand((3, 3), seed=5) + 3.0 * np.eye(3)
_SPD = _GEN @ _GEN.T + np.eye(3)
_CHOL = np.linalg.cholesky(_SPD)
_SINGULAR = np.asarray([[1.0, 2.0], [2.0, 4.0]])
_RHS = _rand((3, 2), seed=6)
_RHS1 = _rand((3,), seed=7)


def _solve_triangular(
    matrix: NDArray[Any],
    rhs: NDArray[Any],
    *,
    lower: bool,
    unit_diagonal: bool = False,
) -> NDArray[Any]:
    result: NDArray[Any] = scipy.linalg.solve_triangular(
        matrix, rhs, lower=lower, unit_diagonal=unit_diagonal
    )
    return result


def _cholesky_solve(factor: NDArray[Any], rhs: NDArray[Any]) -> NDArray[Any]:
    result: NDArray[Any] = scipy.linalg.cho_solve((factor, True), rhs)
    return result


def _lu(array: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    p, lower, upper = scipy.linalg.lu(array)
    return p, lower, upper


def _lu_factor(array: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[np.int64]]:
    lu, pivots = scipy.linalg.lu_factor(array)
    return lu, pivots.astype(np.int64)


def _lu_solve(
    lu: NDArray[Any],
    pivots: NDArray[np.int64],
    rhs: NDArray[Any],
    transpose: bool = False,
) -> NDArray[Any]:
    result: NDArray[Any] = scipy.linalg.lu_solve(
        (lu, pivots), rhs, trans=1 if transpose else 0
    )
    return result


_LU, _PIVOTS = _lu_factor(_GEN)

# Tall full-rank and rank-deficient (third column the sum of the first
# two) matrices, their right-hand sides, and a complex Hermitian matrix.
_TALL = _rand((4, 3), seed=8)
_DEFICIENT = np.column_stack([_TALL[:, :2], _TALL[:, 0] + _TALL[:, 1]])
_TALL_RHS = _rand((4, 2), seed=9)
_CSQ = _crand((3, 3), seed=10)
_HERM = _CSQ @ _CSQ.conj().T + np.eye(3)


def _lstsq(
    a: NDArray[Any], b: NDArray[Any], rcond: Optional[float] = None
) -> NDArray[Any]:
    solution: NDArray[Any] = np.linalg.lstsq(a, b, rcond=rcond)[0]
    return solution


def _pinv(matrix: NDArray[Any], rtol: Optional[float] = None) -> NDArray[Any]:
    # The Array API default cutoff, not NumPy's legacy rcond=1e-15.
    if rtol is None:
        rtol = max(matrix.shape[-2:]) * float(np.finfo(matrix.dtype).eps)
    return np.linalg.pinv(matrix, rtol=rtol)


def _solve_sparse(Amat: csc_matrix, bvec: NDArray[Any]) -> NDArray[Any]:
    dense: NDArray[Any] = Amat.toarray()
    return np.linalg.solve(dense, bvec)


# Fixed weights giving each vector of a factorization a sign: that of its
# weighted sum, nonzero for any vector met here.
_SIGN_WEIGHTS = _rand((8,), seed=11)


def _column_signs(bkd: "ConformanceBackend[Array]", vectors: Array) -> Array:
    # Arithmetic on the comparison, not sign or where: it needs nothing
    # from a backend beyond the operators.
    weights = bkd.asarray(_SIGN_WEIGHTS[: vectors.shape[0]])
    return (weights @ vectors >= 0) * 2.0 - 1.0


def _canonical_eigh(
    bkd: "ConformanceBackend[Array]", result: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    values, vectors = result
    return values, vectors * _column_signs(bkd, vectors)


def _canonical_svd(
    bkd: "ConformanceBackend[Array]", result: Tuple[Array, Array, Array]
) -> Tuple[Array, Array, Array]:
    # Flipping a left singular vector flips its right one.
    left, values, right = result
    signs = _column_signs(bkd, left)
    return left * signs, values, right * signs[:, None]


def _canonical_qr(
    bkd: "ConformanceBackend[Array]", result: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    # Flipping a column of Q flips the matching row of R.
    q, r = result
    signs = _column_signs(bkd, q)
    return q * signs, r * signs[:, None]


def _reductions(
    method: str, reference: Optional[Callable[..., Any]] = None
) -> List[Case]:
    """The axis rows every reduction shares: NumPy's ``None`` (every
    axis, keepdims included), int, negative, tuple and ``()`` (none)."""
    return [
        Case(method, "axis-none", (_A,), reference=reference, differentiable=True),
        Case(method, "axis-0", (_A,), {"axis": 0}, reference=reference,
             differentiable=True),
        Case(method, "axis-neg", (_T3,), {"axis": -1}, reference=reference,
             differentiable=True),
        Case(method, "axis-tuple", (_T3,), {"axis": (0, 2)}, reference=reference,
             differentiable=True),
        Case(method, "keepdims", (_A,), {"axis": 1, "keepdims": True},
             reference=reference),
        Case(method, "keepdims-axis-none", (_A,), {"keepdims": True},
             reference=reference),
        Case(method, "axis-empty", (_A,), {"axis": ()}, reference=reference),
    ]  # fmt: skip


def _logsumexp(
    array: NDArray[Any],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> NDArray[Any]:
    return np.asarray(scipy.special.logsumexp(array, axis=axis, keepdims=keepdims))


def _quantile(array: NDArray[Any], level: float, axis: int = 0) -> NDArray[Any]:
    # The protocol's default axis is 0; NumPy's flattens.
    return np.asarray(np.quantile(array, level, axis=axis))


def _cov(array: NDArray[Any], rowvar: bool = True, ddof: int = 1) -> NDArray[Any]:
    # 2-D for one variable too.
    return np.atleast_2d(np.cov(array, rowvar=rowvar, ddof=ddof))


# Ties (the first occurrence wins) and an input with no rows.
_TIES = np.asarray([[1.0, 3.0, 3.0], [2.0, 2.0, 1.0]])
_EMPTY = np.zeros((0, 3))
# Shape and indexing: singleton axes, a sorted vector with a repeat,
# and values to place in it.
_SINGLETONS = _A[None, :, :, None]
_SORTED = np.asarray([1.0, 2.0, 2.0, 3.0])
_PLACE = np.asarray([2.0, 0.5, 4.0, 3.0])
_UNSORTED = np.asarray([3.0, np.nan, 1.0, 3.0, 2.0])


def _unspecified_contents(bkd: "ConformanceBackend[Array]", result: Array) -> Array:
    # empty's values are arbitrary: compare its shape and dtype only.
    return bkd.zeros(tuple(result.shape), dtype=result.dtype)


def _is_floating(array: NDArray[Any]) -> bool:
    return bool(np.issubdtype(array.dtype, np.floating))


def _is_integer(array: NDArray[Any]) -> bool:
    return bool(np.issubdtype(array.dtype, np.integer))


def _to_float(value: Union[float, NDArray[Any]]) -> float:
    return float(np.asarray(value).reshape(()))


def _to_int(value: Union[int, float, NDArray[Any]]) -> int:
    return int(np.asarray(value).reshape(()))


def _flatten(array: NDArray[Any]) -> NDArray[Any]:
    return array.flatten()


def _argsort(array: NDArray[Any], axis: int = -1) -> NDArray[np.intp]:
    # NumPy's default quicksort is not stable.
    return np.argsort(array, axis=axis, kind="stable")


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
    # A dtype object is not an array, so there is nothing to trace.
    Case("default_dtype", "f64", reference=lambda: np.float64,
         expect=Expect.DTYPE, eager_only=True),
    Case("double_dtype", "f64", reference=lambda: np.float64,
         expect=Expect.DTYPE, eager_only=True),
    Case("complex_dtype", "c128", reference=lambda: np.complex128,
         expect=Expect.DTYPE, eager_only=True),
    Case("int64_dtype", "i64", reference=lambda: np.int64,
         expect=Expect.DTYPE, eager_only=True),
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
    Case("to_bool", "bool", (np.asarray([True]),), reference=lambda x: True,
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
    Case("empty", "default", ((2, 3),), canonical=_unspecified_contents,
         expect=Expect.DEFAULT),
    Case("empty", "index", ((4,),), {"dtype": Role.INDEX},
         canonical=_unspecified_contents, expect=Expect.INDEX),
    Case("eye", "square", (3,), expect=Expect.DEFAULT),
    Case("eye", "rectangular", (2, 3), expect=Expect.DEFAULT),
    Case("eye", "index", (2,), {"dtype": Role.INDEX}, expect=Expect.INDEX),
    Case("arange", "stop", (5,), expect=Expect.INDEX),
    Case("arange", "start-stop", (2, 7), expect=Expect.INDEX),
    Case("arange", "step", (0, 10, 3), expect=Expect.INDEX),
    Case("arange", "float", (0.0, 1.0, 0.25), expect=Expect.DEFAULT),
    Case("arange", "dtype", (4,), {"dtype": Role.DEFAULT},
         expect=Expect.DEFAULT),
    Case("linspace", "default", (0.0, 1.0, 5), expect=Expect.DEFAULT),
    Case("logspace", "default", (0.0, 2.0, 3), expect=Expect.DEFAULT),
    Case("zeros_like", "float", (_A,)),
    Case("zeros_like", "dtype", (_A,), {"dtype": Role.INDEX},
         expect=Expect.INDEX),
    Case("ones_like", "complex", (_C,)),
    Case("ones_like", "dtype", (_A,), {"dtype": Role.COMPLEX},
         expect=Expect.COMPLEX),
    Case("full_like", "float", (_A, 2.5)),
    Case("full_like", "index", (_index(1, 2), 3)),
    Case("array", "float-list", ([[1.0, 2.0], [3.0, 4.0]],),
         reference=lambda x: np.array(x, dtype=np.float64),
         expect=Expect.DEFAULT, eager_only=True),
    Case("array", "int-list", ([1, 2, 3],), expect=Expect.INDEX,
         eager_only=True),
    Case("array", "index", (_index(3, 1),), {"dtype": Role.INDEX},
         expect=Expect.INDEX),
    Case("array", "complex", (_C,)),
    Case("array_type", "class", reference=_none, expect=Expect.ARRAY_TYPE,
         eager_only=True),
    # ---- dtype predicates and host conversion
    Case("is_floating_dtype", "float", (_A,), reference=_is_floating,
         expect=Expect.HOST, eager_only=True),
    Case("is_floating_dtype", "complex", (_C,), reference=_is_floating,
         expect=Expect.HOST, eager_only=True),
    Case("is_floating_dtype", "index", (_index(1),), reference=_is_floating,
         expect=Expect.HOST, eager_only=True),
    Case("is_integer_dtype", "index", (_index(1),), reference=_is_integer,
         expect=Expect.HOST, eager_only=True),
    Case("is_integer_dtype", "float", (_A,), reference=_is_integer,
         expect=Expect.HOST, eager_only=True),
    Case("is_integer_dtype", "bool", (_MASK,), reference=_is_integer,
         expect=Expect.HOST, eager_only=True),
    Case("tracks_gradient", "plain", (_A,), reference=lambda x: False,
         expect=Expect.HOST, eager_only=True),
    Case("to_float", "0d", (np.asarray(2.5),), reference=_to_float,
         expect=Expect.HOST, eager_only=True),
    Case("to_float", "one-element", (np.asarray([[-1.5]]),),
         reference=_to_float, expect=Expect.HOST, eager_only=True),
    Case("to_float", "index", (_index(4),), reference=_to_float,
         expect=Expect.HOST, eager_only=True),
    Case("to_float", "python", (2.5,), reference=_to_float,
         expect=Expect.HOST, eager_only=True),
    Case("to_float", "several", (_A,), raises=ValueError, eager_only=True),
    Case("to_float", "complex", (_C[:1, :1],), raises=TypeError,
         eager_only=True),
    Case("to_int", "index", (_index(7),), reference=_to_int,
         expect=Expect.HOST, eager_only=True),
    # Truncation toward zero, as int().
    Case("to_int", "truncates", (np.asarray(-2.7),), reference=_to_int,
         expect=Expect.HOST, eager_only=True),
    Case("to_int", "python", (3,), reference=_to_int, expect=Expect.HOST,
         eager_only=True),
    Case("to_int", "empty", (np.zeros((0,)),), raises=ValueError,
         eager_only=True),
    Case("to_int", "complex", (_C[:1, :1],), raises=TypeError,
         eager_only=True),
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
    Case("flatten", "matrix", (_A,), reference=_flatten, differentiable=True),
    Case("flatten", "vector", (_RHS1,), reference=_flatten),
    Case("ravel", "3d", (_T3,), differentiable=True),
    Case("squeeze", "singletons", (_SINGLETONS,), differentiable=True),
    Case("transpose", "reverse-3d", (_T3,), differentiable=True),
    Case("transpose", "axes", (_T3, (1, 0, 2)), differentiable=True),
    Case("flip", "every-axis", (_A,), differentiable=True),
    Case("flip", "axis-0", (_A,), {"axis": 0}),
    Case("flip", "axis-tuple", (_T3,), {"axis": (0, 2)}),
    Case("tile", "matrix", (_A, (2, 1)), differentiable=True),
    Case("tile", "short-reps", (_A, (3,))),
    Case("repeat", "axis-none", (_A, 2), differentiable=True),
    Case("repeat", "axis-1", (_A, 3), {"axis": 1}),
    # Eager only: the indices fix the output shapes.
    Case("split", "indices", (np.arange(6.0), _index(2, 4)), eager_only=True),
    Case("split", "axis-1", (_B, _index(1, 3)), {"axis": 1}, eager_only=True),
    Case("hstack", "matrices", ((_A, _POS),), differentiable=True),
    Case("hstack", "vectors", ((_RHS1, _A[0]),)),
    Case("vstack", "matrices", ((_A, _POS),), differentiable=True),
    Case("vstack", "vectors", ((_RHS1, _RHS1),)),
    Case("block", "2x2", ([[_GEN, _RHS], [_RHS.T.copy(), _GEN[:2, :2]]],),
         differentiable=True),
    Case("tril", "default", (_GEN,), differentiable=True),
    Case("tril", "k-1", (_GEN,), {"k": -1}),
    Case("triu", "default", (_GEN,), differentiable=True),
    Case("triu", "k-1", (_GEN,), {"k": 1}),
    Case("tril_indices", "default", (3,), expect=Expect.INDEX),
    Case("tril_indices", "k-1", (4,), {"k": -1}, expect=Expect.INDEX),
    Case("get_diagonal", "default", (_GEN,), reference=np.diagonal,
         differentiable=True),
    Case("get_diagonal", "offset", (_A,), {"offset": 1}, reference=np.diagonal),
    Case("get_diagonal", "axes", (_T3,), {"axis1": 1, "axis2": 2},
         reference=np.diagonal),
    Case("diag", "build", (_RHS1,), differentiable=True),
    Case("diag", "extract", (_GEN,), differentiable=True),
    Case("diag", "offset", (_RHS1,), {"k": 1}),
    Case("nonzero", "matrix", (_MASK,), expect=Expect.INDEX, eager_only=True),
    Case("unique", "repeats", (_TIES,), eager_only=True),
    Case("unique", "nan", (_UNSORTED,), eager_only=True),
    Case("sort", "last-axis", (_A,), differentiable=True),
    Case("sort", "axis-0", (_A,), {"axis": 0}),
    Case("sort", "nan", (_UNSORTED,)),
    Case("sort", "complex", (_C,), raises=TypeError),
    Case("argsort", "ties-stable", (_TIES,), reference=_argsort,
         expect=Expect.INDEX),
    Case("argsort", "axis-0", (_TIES,), {"axis": 0}, reference=_argsort,
         expect=Expect.INDEX),
    Case("argsort", "nan", (_UNSORTED,), reference=_argsort,
         expect=Expect.INDEX),
    Case("argsort", "complex", (_C,), raises=TypeError),
    Case("searchsorted", "left", (_SORTED, _PLACE), expect=Expect.INDEX),
    Case("searchsorted", "right", (_SORTED, _PLACE), {"side": "right"},
         expect=Expect.INDEX),
    Case("atleast_1d", "0d", (np.asarray(2.0),)),
    Case("atleast_1d", "vector", (_RHS1,)),
    Case("atleast_2d", "vector", (_RHS1,), differentiable=True),
    Case("atleast_2d", "matrix", (_A,)),
    Case("meshgrid", "xy", (_RHS1, _A[0, :2])),
    Case("meshgrid", "ij", (_RHS1, _A[0, :2]), {"indexing": "ij"}),
    Case("copy", "matrix", (_A,), differentiable=True),
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
    Case("sum", "complex", (_C,), {"axis": 0}),
    *_reductions("mean"),
    Case("mean", "complex", (_C,), {"axis": 0}),
    Case("prod", "axis-none", (_POS,), differentiable=True),
    Case("prod", "axis-1", (_POS,), {"axis": 1}, differentiable=True),
    Case("prod", "axis-tuple", (_T3,), {"axis": (0, 2)}, differentiable=True),
    Case("prod", "keepdims", (_POS,), {"axis": 0, "keepdims": True}),
    Case("prod", "keepdims-axis-none", (_POS,), {"keepdims": True}),
    Case("prod", "axis-empty", (_POS,), {"axis": ()}),
    *_reductions("max"),
    Case("max", "nan", (_NAN,)),
    Case("max", "empty-other-axis", (_EMPTY,), {"axis": 1}),
    Case("max", "empty", (_EMPTY,), {"axis": 0}, raises=ValueError),
    Case("max", "complex", (_C,), raises=TypeError),
    *_reductions("min"),
    Case("min", "nan", (_NAN,)),
    Case("min", "empty", (_EMPTY,), raises=ValueError),
    Case("min", "complex", (_C,), {"axis": 0}, raises=TypeError),
    Case("argmax", "axis-none", (_A,), expect=Expect.INDEX),
    Case("argmax", "axis-0", (_A,), {"axis": 0}, expect=Expect.INDEX),
    Case("argmax", "axis-neg", (_T3,), {"axis": -1}, expect=Expect.INDEX),
    Case("argmax", "ties", (_TIES,), {"axis": 1}, expect=Expect.INDEX),
    Case("argmax", "nan", (_NAN,), expect=Expect.INDEX),
    Case("argmax", "empty", (_EMPTY,), {"axis": 0}, raises=ValueError),
    Case("argmax", "complex", (_C,), raises=TypeError),
    Case("argmin", "axis-none", (_A,), expect=Expect.INDEX),
    Case("argmin", "axis-1", (_A,), {"axis": 1}, expect=Expect.INDEX),
    Case("argmin", "ties", (_TIES,), {"axis": 0}, expect=Expect.INDEX),
    Case("argmin", "nan", (_NAN,), expect=Expect.INDEX),
    Case("argmin", "empty", (_EMPTY,), raises=ValueError),
    Case("argmin", "complex", (_C,), {"axis": 0}, raises=TypeError),
    *_reductions("var"),
    Case("var", "ddof", (_A,), {"axis": 1, "ddof": 1}, differentiable=True),
    Case("var", "complex", (_C,), {"axis": 0}, expect=Expect.DEFAULT),
    *_reductions("std"),
    Case("std", "ddof", (_A,), {"axis": 1, "ddof": 1}, differentiable=True),
    Case("std", "complex", (_C,), {"axis": 0}, expect=Expect.DEFAULT),
    *_reductions("logsumexp", reference=_logsumexp),
    Case("logsumexp", "large", (np.asarray([1000.0, 1000.0]),),
         reference=_logsumexp),
    Case("cumsum", "axis-none", (_A,), differentiable=True),
    Case("cumsum", "axis-1", (_A,), {"axis": 1}, differentiable=True),
    Case("cumsum", "complex", (_C,), {"axis": 0}),
    Case("quantile", "axis-0", (_A, 0.3), reference=_quantile,
         differentiable=True),
    Case("quantile", "axis-1", (_A, 0.7), {"axis": 1}, reference=_quantile,
         differentiable=True),
    Case("quantile", "vector", (_RHS1, 0.5), reference=_quantile),
    Case("quantile", "level-0", (_A, 0.0), reference=_quantile),
    Case("quantile", "level-1", (_A, 1.0), {"axis": 1}, reference=_quantile),
    Case("cov", "rows", (_A,), reference=_cov, differentiable=True),
    Case("cov", "columns", (_B,), {"rowvar": False}, reference=_cov,
         differentiable=True),
    Case("cov", "ddof-0", (_A,), {"ddof": 0}, reference=_cov),
    Case("cov", "one-variable", (_A[0],), reference=_cov),
    Case("diff", "default", (_A,), differentiable=True),
    Case("diff", "n-2", (_B,), {"n": 2}, differentiable=True),
    Case("diff", "axis-0", (_B,), {"axis": 0}, differentiable=True),
    Case("any", "axis-none", (_A > 0.5,), expect=Expect.BOOL),
    Case("any", "axis-tuple", (_T3 > 0.5,), {"axis": (1, 2)},
         expect=Expect.BOOL),
    Case("any", "float-input", (_A,), {"axis": 0}, expect=Expect.BOOL),
    Case("any", "integer-input", (_index(0, 3, 0),), expect=Expect.BOOL),
    Case("any", "axis-empty", (_A > 0.5,), {"axis": ()}, expect=Expect.BOOL),
    Case("any", "0d", (np.asarray(0.0),), expect=Expect.BOOL),
    Case("any", "empty-axis-0", (_EMPTY,), {"axis": 0}, expect=Expect.BOOL),
    Case("all", "keepdims", (_MASK,), {"axis": 1, "keepdims": True},
         expect=Expect.BOOL),
    Case("all", "keepdims-axis-none", (_MASK,), {"keepdims": True},
         expect=Expect.BOOL),
    Case("all", "axis-neg", (_MASK,), {"axis": -1}, expect=Expect.BOOL),
    Case("all", "axis-tuple-neg", (_T3 > -0.5,), {"axis": (-1, 0)},
         expect=Expect.BOOL),
    Case("all", "empty", (_EMPTY,), expect=Expect.BOOL),
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
    # ---- linear algebra: solves and factorizations
    Case("solve", "matrix-rhs", (_GEN, _RHS), differentiable=True),
    Case("solve", "vector-rhs", (_GEN, _RHS1), differentiable=True),
    Case("solve", "singular", (_SINGULAR, _RHS[:2]), raises=LinAlgError),
    Case("solve_triangular", "lower", (_CHOL, _RHS), {"lower": True},
         reference=_solve_triangular, differentiable=True),
    Case("solve_triangular", "upper", (_CHOL.T.copy(), _RHS), {"lower": False},
         reference=_solve_triangular, differentiable=True),
    Case("solve_triangular", "unit-diagonal", (_CHOL, _RHS),
         {"lower": True, "unit_diagonal": True}, reference=_solve_triangular),
    Case("solve_triangular", "vector-rhs", (_CHOL, _RHS1), {"lower": True},
         reference=_solve_triangular, differentiable=True),
    Case("cholesky", "spd", (_SPD,)),
    Case("cholesky", "not-positive-definite", (-_SPD,), raises=LinAlgError),
    Case("cholesky_solve", "matrix-rhs", (_CHOL, _RHS), reference=_cholesky_solve),
    Case("cholesky_solve", "vector-rhs", (_CHOL, _RHS1),
         reference=_cholesky_solve),
    Case("lu", "general", (_GEN,), reference=_lu),
    Case("lu_factor", "general", (_GEN,), reference=_lu_factor,
         parts=(Expect.INPUT, Expect.INDEX)),
    Case("lu_solve", "matrix-rhs", (_LU, _PIVOTS, _RHS), reference=_lu_solve),
    Case("lu_solve", "vector-rhs", (_LU, _PIVOTS, _RHS1), reference=_lu_solve),
    Case("lu_solve", "transpose", (_LU, _PIVOTS, _RHS), {"transpose": True},
         reference=_lu_solve),
    Case("inv", "general", (_GEN,), differentiable=True),
    Case("inv", "singular", (_SINGULAR,), raises=LinAlgError),
    Case("det", "general", (_GEN,), differentiable=True),
    Case("slogdet", "general", (_GEN,)),
    Case("solve_sparse", "vector-rhs", kwargs={"Amat": Sparse(_GEN), "bvec": _RHS1},
         reference=_solve_sparse, expect=Expect.DEFAULT, eager_only=True,
         optional=True),
    Case("solve_sparse", "matrix-rhs", kwargs={"Amat": Sparse(_GEN), "bvec": _RHS},
         reference=_solve_sparse, expect=Expect.DEFAULT, eager_only=True,
         optional=True),
    # ---- linear algebra: decompositions, least squares, norms, products
    Case("qr", "reduced", (_TALL,), canonical=_canonical_qr),
    Case("qr", "complete", (_TALL,), {"mode": "complete"},
         canonical=_canonical_qr),
    Case("svd", "square", (_GEN,), canonical=_canonical_svd),
    Case("svd", "tall-reduced", (_TALL,), {"full_matrices": False},
         canonical=_canonical_svd),
    Case("eigh", "spd", (_SPD,), canonical=_canonical_eigh),
    # Not differentiable as a row: eigvalsh reads one triangle, while the
    # autodiff libraries give the gradient of a symmetric input.
    Case("eigvalsh", "spd", (_SPD,)),
    Case("eigvalsh", "hermitian", (_HERM,), expect=Expect.DEFAULT),
    Case("pinv", "tall", (_TALL,), reference=_pinv, differentiable=True),
    # 8e-16 is above the default cutoff 2 * eps, below NumPy's legacy 1e-15.
    Case("pinv", "default-cutoff", (np.diag([1.0, 8e-16]),), reference=_pinv),
    Case("pinv", "rtol", (np.diag([1.0, 1e-3]),), {"rtol": 1e-2},
         reference=_pinv),
    # Eager only: torch's solver has an output of data-dependent shape, so
    # it does not compile; a compiled kernel solves through pinv or qr.
    Case("lstsq", "vector-rhs", (_TALL, _TALL_RHS[:, 0]), reference=_lstsq,
         eager_only=True, differentiable=True),
    Case("lstsq", "matrix-rhs", (_TALL, _TALL_RHS), reference=_lstsq,
         eager_only=True, differentiable=True),
    Case("lstsq", "rank-deficient", (_DEFICIENT, _TALL_RHS), reference=_lstsq,
         eager_only=True),
    Case("cond", "2-norm", (_GEN,), differentiable=True),
    Case("cond", "fro", (_GEN, "fro"), differentiable=True),
    Case("norm", "vector", (_RHS1,), differentiable=True),
    Case("norm", "matrix-fro", (_GEN,), differentiable=True),
    Case("norm", "axis-0-ord-1", (_A,), {"ord": 1, "axis": 0},
         differentiable=True),
    Case("norm", "nuc", (_GEN,), {"ord": "nuc"}),
    Case("norm", "vector-inf", (_RHS1,), {"ord": float("inf")}),
    Case("norm", "keepdims", (_A,), {"keepdims": True}),
    Case("norm", "axis-tuple", (_T3,), {"axis": (1, 2)}, differentiable=True),
    Case("norm", "complex", (_C[0],), expect=Expect.DEFAULT),
    Case("trace", "matrix", (_GEN,), reference=np.linalg.trace,
         differentiable=True),
    Case("trace", "batched", (_T3[:, :, :3],), reference=np.linalg.trace,
         differentiable=True),
    Case("multidot", "matrices", ((_GEN, _TALL.T, _TALL),),
         reference=np.linalg.multi_dot, differentiable=True),
    Case("multidot", "vector-ends", ((_GEN[0], _GEN, _GEN[1]),),
         reference=np.linalg.multi_dot, differentiable=True),
    Case("kron", "matrices", (_GEN[:2, :2], _TALL), differentiable=True),
    Case("outer", "vectors", (_A[0], _B[0]), differentiable=True),
    Case("outer", "flattens", (_A, _GEN[0]), differentiable=True),
]  # fmt: skip

# Protocol methods without a row. Shrink-only: see the module docstring.
NOT_YET_COVERED = frozenset(
    {
        "abs", "all_array", "all_bool", "allclose", "any_array", "any_bool",
        "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2",
        "arctanh", "cdist", "ceil", "clip", "conj", "cos", "cosh",
        "digamma", "equal", "erf", "erfinv", "expm1", "floor", "floor_divide",
        "gammaln", "isfinite", "isinf", "isnan", "log", "log1p", "log_ndtr",
        "maximum", "minimum", "ndtr", "round", "sign", "sin", "sinh", "tanh",
    }
)  # fmt: skip

# Methods with a row that JaxBkd does not implement yet. With
# NOT_YET_COVERED they make NOT_YET_ON_JAX, the protocol methods JaxBkd
# lacks; its length is the conformance_not_on_jax baseline, so a row can
# land before its JAX implementation but a new protocol method cannot
# arrive without one.
ROWS_NOT_YET_ON_JAX: frozenset[str] = frozenset()
NOT_YET_ON_JAX = NOT_YET_COVERED | ROWS_NOT_YET_ON_JAX

# Protocol methods tested by a column over every row rather than by rows
# of their own: jit compiles each traceable row (row_kernel).
COVERED_BY_COLUMNS = frozenset({"jit"})


# ---- running a row


@runtime_checkable
class ConformanceBackend(Protocol[Array]):
    """What the runner itself calls on a backend besides the method under
    test, so a backend still growing toward ``Backend`` (``JaxBkd``) can
    run the table. Dtypes are ``Any`` as in ``Backend``."""

    def default_dtype(self) -> Any: ...

    def complex_dtype(self) -> Any: ...

    def int64_dtype(self) -> Any: ...

    def asarray(self, array: ArrayLike, dtype: Optional[Any] = None) -> Array: ...

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Array: ...

    def to_numpy(self, array: Array) -> NDArray[Any]: ...

    def assert_allclose(
        self, actual: Array, desired: Array, rtol: float = 1e-7, atol: float = 0
    ) -> None: ...

    def jit(
        self,
        kernel: Callable[..., KernelResult],
        static_argnames: Sequence[str] = (),
        dynamic: bool = False,
    ) -> Callable[..., KernelResult]: ...


def _bool_dtype(bkd: ConformanceBackend[Array]) -> Any:
    # A comparison is the backend-neutral way to name the boolean dtype.
    return (bkd.zeros((1,)) > 0).dtype


def _backend_dtype(bkd: ConformanceBackend[Array], role: Role) -> Any:
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


def _to_backend(bkd: ConformanceBackend[Array], value: Any) -> Any:
    """``value`` with NumPy arrays (also inside tuples and lists) on
    ``bkd`` in the matching role's dtype, and roles resolved to ``bkd``
    dtypes."""
    if isinstance(value, Role):
        return _backend_dtype(bkd, value)
    if isinstance(value, Sparse):
        return value.to_scipy()
    if isinstance(value, np.ndarray):
        role = _ROLE_OF_NUMPY[value.dtype]
        return bkd.asarray(value, dtype=_backend_dtype(bkd, role))
    if isinstance(value, (tuple, list)):
        return type(value)(_to_backend(bkd, v) for v in value)
    return value


def _to_reference(value: Any) -> Any:
    if isinstance(value, Sparse):
        return value.to_scipy()
    return _NUMPY_DTYPES[value] if isinstance(value, Role) else value


def _reference(case: Case) -> Any:
    # The NumPy function of the same name, in numpy or numpy.linalg.
    function = (
        case.reference
        or getattr(np, case.method, None)
        or getattr(np.linalg, case.method)
    )
    args = [_to_reference(a) for a in case.args]
    kwargs = {k: _to_reference(v) for k, v in case.kwargs.items()}
    return function(*args, **kwargs)


def _arrays(values: Union[Tuple[Any, ...], List[Any]]) -> List[Any]:
    """The backend arrays among ``values``, also inside tuples and
    lists."""
    found: List[Any] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            found += _arrays(value)
        elif hasattr(value, "dtype") and hasattr(value, "shape"):
            found.append(value)
    return found


def _role_dtype(
    bkd: ConformanceBackend[Array], expect: Expect, args: Tuple[Any, ...]
) -> Any:
    if expect is Expect.INPUT:
        bool_dtype = _bool_dtype(bkd)
        return next(a.dtype for a in _arrays(args) if a.dtype != bool_dtype)
    return _backend_dtype(bkd, Role(expect.value))


def _expected_dtype(
    bkd: ConformanceBackend[Array], case: Case, args: Tuple[Any, ...]
) -> Any:
    """The expected dtype of the result, or a tuple of them, one per
    element, for a row with ``parts``."""
    if case.parts:
        return tuple(_role_dtype(bkd, part, args) for part in case.parts)
    return _role_dtype(bkd, case.expect, args)


def _assert_matches(
    bkd: ConformanceBackend[Array], actual: Any, expected: Any, dtype: Any
) -> None:
    """``actual`` equals ``expected`` (NumPy, converted to ``dtype`` on
    ``bkd``) in shape, dtype and values; tuples and lists elementwise,
    with one dtype per element when ``dtype`` is a tuple."""
    if isinstance(expected, (tuple, list)):
        # A list where NumPy gives one, a tuple (any named tuple) otherwise.
        kind = list if isinstance(expected, list) else tuple
        assert isinstance(actual, kind), (type(actual), type(expected))
        assert len(actual) == len(expected), (len(actual), len(expected))
        dtypes = dtype if isinstance(dtype, tuple) else (dtype,) * len(expected)
        assert len(dtypes) == len(expected), (len(dtypes), len(expected))
        for a, e, d in zip(actual, expected, dtypes):
            _assert_matches(bkd, a, e, d)
        return
    # A copy: a NumPy view with negative strides (flip) cannot become a
    # torch tensor.
    expected = np.array(expected)
    tol = 0.0
    if np.issubdtype(expected.dtype, np.inexact):
        tol = 10 * float(np.finfo(expected.dtype).eps)
    desired = bkd.asarray(expected, dtype=dtype)
    bkd.assert_allclose(actual, desired, rtol=tol, atol=tol)


def _assert_host_matches(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected), (type(actual), type(expected))
    np.testing.assert_array_equal(actual, expected, strict=True)


def _assert_dtype_object(
    bkd: ConformanceBackend[Array], actual: Any, expected: Any
) -> None:
    created = bkd.to_numpy(bkd.zeros((1,), dtype=actual)).dtype
    assert created == np.dtype(expected), (created, expected)


def prepare(
    bkd: ConformanceBackend[Array], case: Case
) -> Tuple[Callable[..., Any], Tuple[Any, ...]]:
    """``case``'s method on ``bkd`` with its keyword arguments bound, and
    its positional arguments converted to ``bkd``."""
    args = tuple(_to_backend(bkd, a) for a in case.args)
    kwargs = {k: _to_backend(bkd, v) for k, v in case.kwargs.items()}
    bound: Callable[..., Any] = getattr(bkd, case.method)

    def method(*positional: Any) -> Any:
        return bound(*positional, **kwargs)

    return method, args


def run_case(
    bkd: ConformanceBackend[Array],
    case: Case,
    call: Optional[Callable[[Callable[..., Any], Tuple[Any, ...]], Any]] = None,
) -> Any:
    """Run ``case`` on ``bkd`` and check it; returns the result.

    ``call(method, args)`` invokes the bound method (keyword arguments
    already bound) on the converted positional arguments, so a backend's
    own test can run it compiled.
    """
    method, args = prepare(bkd, case)
    invoke = call or (lambda fn, values: fn(*values))
    if case.raises is not None:
        with pytest.raises(case.raises):
            invoke(method, args)
        return None
    try:
        result = invoke(method, args)
    except NotImplementedError:
        if not case.optional:
            raise
        pytest.skip(f"{type(bkd).__name__} declines {case.method}")
    expected = _reference(case)
    if case.canonical is not None:
        result = case.canonical(bkd, result)
        expected = case.canonical(NumpyBkd(), expected)
    if case.expect is Expect.NONE:
        assert result is None
    elif case.expect is Expect.HOST:
        _assert_host_matches(result, expected)
    elif case.expect is Expect.ARRAY_TYPE:
        created = bkd.zeros((1,))
        assert isinstance(result, type) and isinstance(created, result), (
            result,
            type(created),
        )
    elif case.expect is Expect.DTYPE:
        _assert_dtype_object(bkd, result, expected)
    else:
        _assert_matches(bkd, result, expected, _expected_dtype(bkd, case, args))
    # The inputs are unchanged: compare each with a fresh conversion.
    fresh = _arrays(tuple(_to_backend(bkd, a) for a in case.args))
    for array, original in zip(_arrays(args), fresh):
        bkd.assert_allclose(array, original, rtol=0.0, atol=0.0)
    return result


def _substitute(
    bkd: ConformanceBackend[Array], value: Any, arrays: Iterator[Array]
) -> Any:
    """``value`` with its NumPy arrays (also inside tuples and lists)
    replaced, in order, by ``arrays`` and its roles resolved to ``bkd``
    dtypes."""
    if isinstance(value, Role):
        return _backend_dtype(bkd, value)
    if isinstance(value, np.ndarray):
        return next(arrays)
    if isinstance(value, (tuple, list)):
        return type(value)(_substitute(bkd, v, arrays) for v in value)
    return value


def row_kernel(*arrays: Array, bkd: ConformanceBackend[Array], case_index: int) -> Any:
    """The row ``CASES[case_index]`` as a kernel for ``bkd.jit``: its
    arrays are the traced arguments, in the order they appear in the
    row's positional arguments; everything else comes from the row, which
    ``case_index`` (static) selects."""
    case = CASES[case_index]
    remaining = iter(arrays)
    args = tuple(_substitute(bkd, a, remaining) for a in case.args)
    method: Callable[..., Any] = getattr(bkd, case.method)
    return method(*args, **{k: _to_backend(bkd, v) for k, v in case.kwargs.items()})


def jit_call(
    bkd: ConformanceBackend[Array], case: Case
) -> Callable[[Callable[..., Any], Tuple[Any, ...]], Any]:
    """A ``call`` hook for :func:`run_case` that runs ``case`` through
    ``bkd.jit(row_kernel)`` instead of the method it is given."""
    # By identity: rows hold NumPy arrays, which == cannot compare.
    index = next(i for i, row in enumerate(CASES) if row is case)
    compiled = bkd.jit(row_kernel, static_argnames=("case_index",))

    def call(method: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
        return compiled(*_arrays(args), case_index=index)

    return call


def _floating(bkd: AutodiffBackend[Array], args: Tuple[Any, ...]) -> List[Array]:
    """The floating arrays among ``args``, also inside tuples, in order."""
    return [a for a in _arrays(args) if bkd.is_floating_dtype(a)]


def _replace_floating(
    bkd: AutodiffBackend[Array], value: Any, leaves: Iterator[Array]
) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(_replace_floating(bkd, v, leaves) for v in value)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return next(leaves) if bkd.is_floating_dtype(value) else value
    return value


def check_jacobian(bkd: AutodiffBackend[Array], case: Case, step: float = 1e-3) -> None:
    """Check ``bkd.jacobian`` of a differentiable row against central
    differences.

    The row's floating inputs are flattened into one vector ``x``; along
    a fixed random direction ``v``, the Jacobian-vector product is
    compared with ``(f(x + h v) - f(x - h v)) / 2h`` at ``h = step`` and
    ``step / 2``. Central differences are exact up to rounding for
    functions linear or quadratic in ``x``, so the row passes if the
    error is at rounding level, or if it falls by about 4 when the step
    halves (second order), which shows a correct Jacobian whatever the
    function's scale. Only the backend and NumPy are used, so the check
    stays inside the backend layer.
    """
    method, args = prepare(bkd, case)
    leaves = _floating(bkd, args)
    assert leaves, "a differentiable row needs a floating input"
    shapes = [tuple(leaf.shape) for leaf in leaves]
    sizes = [int(np.prod(shape)) for shape in shapes]
    point = bkd.concatenate([bkd.reshape(leaf, (-1,)) for leaf in leaves])

    def function(flat: Array) -> Array:
        pieces, start = [], 0
        for shape, size in zip(shapes, sizes):
            pieces.append(bkd.reshape(flat[start : start + size], shape))
            start += size
        result = method(*_replace_floating(bkd, args, iter(pieces)))
        return bkd.reshape(result, (-1,))

    direction = np.random.default_rng(7).standard_normal(sum(sizes))
    vec = bkd.asarray(direction / np.linalg.norm(direction))
    jvp = bkd.jacobian(function, point) @ vec
    scale = max(1.0, float(np.max(np.abs(bkd.to_numpy(jvp)), initial=0.0)))
    errors = []
    for h in (step, step / 2):
        central = (function(point + h * vec) - function(point - h * vec)) / (2 * h)
        diff = bkd.to_numpy(central - jvp)
        errors.append(float(np.max(np.abs(diff), initial=0.0)) / scale)
    rounding = 1e-9
    if errors[0] < rounding and errors[1] < rounding:
        return
    ratio = errors[0] / errors[1]
    assert 3.0 < ratio < 5.0, (
        f"central-difference errors {errors[0]:.2e} at h={step:g} and "
        f"{errors[1]:.2e} at h={step / 2:g} (ratio {ratio:.2f}, expected 4): "
        "the Jacobian does not match the function"
    )


def case_id(case: Case) -> str:
    """The pytest id of ``case``."""
    return f"{case.method}-{case.id}"
