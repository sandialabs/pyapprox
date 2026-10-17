r"""A dense linear-algebra stack generic over the computational backend.

Vectors are backend arrays of shape ``(size, ncols)``. Every operation
is a backend array expression, so a torch graph through the operands is
preserved; for that reason ``out`` is ignored and every result is a new
array (writing in place would break automatic differentiation). Meant
for small dense problems and as a verification oracle, not for speed.
"""

import warnings
from typing import Any, Generic, Optional, Sequence

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators.algebra import Transposed
from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    OperatorProperties,
    RowSetProtocol,
)
from pyapprox.util.linalg.operators.solvers import (
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
    require_format,
    require_square,
)

_SYMMETRY_RTOL = 1e-12


def _all_finite(x: Array, bkd: Backend[Array]) -> bool:
    nentries = 1
    for extent in x.shape:
        nentries *= int(extent)
    return bkd.to_int(bkd.sum(bkd.isfinite(x))) == nentries


def _require_backend_dtype(
    descriptor: SpaceDescriptor, bkd: Backend[Array], what: str
) -> Any:
    """Return the backend's configured dtype, refusing a space of another.

    The dtype name is read from a one-element array, so any backend that
    reports its default dtype works, whatever its dtype objects are.
    """
    dtype = bkd.default_dtype()
    name = bkd.to_numpy(bkd.zeros((1,), dtype=dtype)).dtype.name
    if descriptor.dtype != name:
        raise ValueError(
            f"{what}: descriptor dtype {descriptor.dtype!r} differs from the backend's "
            f"configured dtype {name!r}; configure the backend or change the descriptor"
        )
    return dtype


class BackendVectorSpace(Generic[Array]):
    """Blocks of vectors stored as backend arrays of shape ``(size, ncols)``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.VectorSpaceProtocol`
    with backend arrays as both the vector and the reduction type.
    ``out`` is accepted and ignored, so callers must use the returned
    value. The aliasing rule is still enforced, by identity, so code
    tested here does not break on a stack that writes in place.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The space of the vectors. Its dtype must be the backend's
        configured dtype (``bkd.default_dtype()``).
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If the descriptor's dtype is not the backend's. From any operation,
        if a vector is not 2-D with ``descriptor.size`` rows, or if ``out``
        is an operand it may not alias.
    """

    def __init__(self, descriptor: SpaceDescriptor, bkd: Backend[Array]) -> None:
        self._dtype: Any = _require_backend_dtype(descriptor, bkd, "BackendVectorSpace")
        self._layout = descriptor
        self._bkd = bkd

    def descriptor(self) -> SpaceDescriptor:
        return self._layout

    def _check(self, x: Array, name: str) -> None:
        if x.ndim != 2 or x.shape[0] != self._layout.size:
            raise ValueError(
                f"{name} must have shape ({self._layout.size}, ncols), "
                f"got {tuple(x.shape)}"
            )

    @staticmethod
    def _check_out(out: Optional[Array], others: tuple[Array, ...]) -> None:
        """Refuse an ``out`` that is one of ``others``, the operands that
        are not accumulated into."""
        if out is not None and any(o is out for o in others):
            raise ValueError("out is an operand it may not alias")

    def zeros(self, ncols: int = 1) -> Array:
        return self._bkd.zeros((self._layout.size, ncols), dtype=self._dtype)

    def zeros_like(self, x: Array) -> Array:
        self._check(x, "x")
        return self._bkd.zeros((self._layout.size, x.shape[1]), dtype=self._dtype)

    def ncols(self, x: Array) -> int:
        self._check(x, "x")
        return int(x.shape[1])

    def copy(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        self._check_out(out, (x,))
        return self._bkd.copy(x)

    def scale(self, a: float, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        self._check_out(out, ())
        return a * x

    def axpby(
        self, a: float, x: Array, b: float, y: Array, out: Optional[Array] = None
    ) -> Array:
        self._check(x, "x")
        self._check(y, "y")
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {tuple(x.shape)} and "
                f"{tuple(y.shape)}"
            )
        self._check_out(out, (x,))
        return a * x + b * y

    def dot(self, x: Array, y: Array) -> Array:
        self._check(x, "x")
        self._check(y, "y")
        return self._bkd.dot(x.T, y)

    def norm(self, x: Array) -> Array:
        self._check(x, "x")
        return self._bkd.sqrt(self._bkd.sum(x * x, axis=0, keepdims=True))

    def divide_entries(self, x: Array, d: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        self._check(d, "d")
        if d.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {tuple(d.shape)}")
        self._check_out(out, (d,))
        return x / d

    def all_finite(self, x: Array) -> bool:
        self._check(x, "x")
        return _all_finite(x, self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._layout!r})"


class BackendRowSet(Generic[Array]):
    """A set of rows of a space, applied to backend blocks by masking.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.RowSetProtocol` for
    backend blocks of shape ``(size, ncols)``. Every operation multiplies
    by a constant 0/1 column mask, so a torch graph through the operands
    is preserved; ``out`` is ignored, as in :class:`BackendVectorSpace`,
    and the aliasing rule is enforced by identity.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The space whose rows are selected. Its dtype must be the
        backend's configured dtype.
    rows : sequence of int
        Row indices; duplicates are removed and order is ignored.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If an index is outside ``[0, descriptor.size)`` or the descriptor's dtype
        is not the backend's.
    """

    def __init__(
        self, descriptor: SpaceDescriptor, rows: Sequence[int], bkd: Backend[Array]
    ) -> None:
        dtype = _require_backend_dtype(descriptor, bkd, "BackendRowSet")
        indices = tuple(sorted({int(row) for row in rows}))
        if indices and (indices[0] < 0 or indices[-1] >= descriptor.size):
            raise ValueError(
                f"row indices must lie in [0, {descriptor.size}), got "
                f"[{indices[0]}, {indices[-1]}]"
            )
        self._layout = descriptor
        self._indices = indices
        self._bkd = bkd
        # The masks are constant data, built once from plain integers.
        in_set = set(indices)
        keep = [[0.0] if row in in_set else [1.0] for row in range(descriptor.size)]
        self._keep = bkd.asarray(keep, dtype=dtype)
        self._take = 1.0 - self._keep

    def descriptor(self) -> SpaceDescriptor:
        return self._layout

    def count(self) -> int:
        return len(self._indices)

    def indices(self) -> tuple[int, ...]:
        """The sorted row indices."""
        return self._indices

    def keep_mask(self) -> Array:
        """A ``(size, 1)`` column: 0 on the rows in the set, 1 elsewhere."""
        return self._keep

    def _check(self, y: Array, out: Optional[Array], others: tuple[Array, ...]) -> None:
        for name, block in (("y", y), *(("x", other) for other in others)):
            if block.ndim != 2 or block.shape[0] != self._layout.size:
                raise ValueError(
                    f"{name} must have shape ({self._layout.size}, ncols), "
                    f"got {tuple(block.shape)}"
                )
        if out is not None and any(other is out for other in others):
            raise ValueError("out is an operand it may not alias")

    def zero_rows(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, out, ())
        return x * self._keep

    def replace_rows(self, y: Array, x: Array, out: Optional[Array] = None) -> Array:
        self._check(y, out, (x,))
        return y * self._keep + x * self._take

    def add_rows(self, y: Array, x: Array, out: Optional[Array] = None) -> Array:
        self._check(y, out, (x,))
        return y + x * self._take

    def difference(self, other: "BackendRowSet[Array]") -> "BackendRowSet[Array]":
        self._layout.require_same(other.descriptor(), "row set to subtract")
        remaining = sorted(set(self._indices) - set(other.indices()))
        return BackendRowSet(self._layout, remaining, self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._layout!r}, count={self.count()})"


def _backend_keep_mask(rows: RowSetProtocol[Any]) -> Any:
    if not isinstance(rows, BackendRowSet):
        raise TypeError(
            f"DenseFormat row edits need a BackendRowSet, got {type(rows).__name__}"
        )
    return rows.keep_mask()


class DenseFormat(Generic[Array]):
    """Dense backend storage, the format of :class:`BackendDenseOperator`.

    Parameters
    ----------
    matrix : Array
        The stored matrix.
    bkd : Backend[Array]
        Computational backend, for the row edits.
    """

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._bkd = bkd

    def matrix(self) -> Array:
        """The stored matrix. Not to be modified: operators are immutable."""
        return self._matrix

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, "DenseFormat[Array]"]]
    ) -> "DenseFormat[Array]":
        total = coefficient * self._matrix
        for c, other in others:
            total = total + c * other._matrix
        return DenseFormat(total, self._bkd)

    def transposed(self) -> "DenseFormat[Array]":
        return DenseFormat(self._matrix.T, self._bkd)

    def rows_replaced(self, rows: RowSetProtocol[Any]) -> "DenseFormat[Array]":
        keep = _backend_keep_mask(rows)
        n = self._matrix.shape[0]
        identity_rows = self._bkd.diag(self._bkd.reshape(1.0 - keep, (n,)))
        return DenseFormat(self._matrix * keep + identity_rows, self._bkd)

    def rows_zeroed(self, rows: RowSetProtocol[Any]) -> "DenseFormat[Array]":
        return DenseFormat(self._matrix * _backend_keep_mask(rows), self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self._matrix.shape)})"


class BackendDenseOperator(Generic[Array]):
    """A linear operator stored as a dense backend array.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.DiagonalCapableProtocol`
    with backend arrays as vectors and :class:`DenseFormat` as its format.
    ``apply`` is a backend product, so a torch graph through the input
    (and through the matrix, if it carries one) is preserved; ``out`` is
    ignored, as in :class:`BackendVectorSpace`.

    Parameters
    ----------
    matrix : Array
        Shape ``(range.size, domain.size)``.
    domain : SpaceDescriptor
        The space of the input of ``apply``. Its dtype, and the range's,
        must be the backend's configured dtype.
    bkd : Backend[Array]
        Computational backend.
    range : SpaceDescriptor, optional
        The space of the output of ``apply``. Defaults to ``domain``.
    properties : OperatorProperties, optional
        Declared properties. ``symmetric``, ``diagonal`` and ``identity``
        are verified; ``spd`` is not (that would cost a factorization).

    Raises
    ------
    ValueError
        If the shape disagrees with the descriptors, a descriptor's dtype is not
        the backend's,
        a symmetric operator's domain and range differ, or a verified
        property does not hold.
    """

    def __init__(
        self,
        matrix: Array,
        domain: SpaceDescriptor,
        bkd: Backend[Array],
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        self._domain = domain
        self._range = domain if range is None else range
        self._properties = OperatorProperties() if properties is None else properties
        self._bkd = bkd
        self._dtype: Any = _require_backend_dtype(
            self._domain, bkd, "BackendDenseOperator"
        )
        _require_backend_dtype(self._range, bkd, "BackendDenseOperator")
        shape = tuple(matrix.shape)
        if shape != (self._range.size, self._domain.size):
            raise ValueError(
                f"matrix must have shape (range.size, domain.size) = "
                f"({self._range.size}, {self._domain.size}), got {shape}"
            )
        if self._properties.symmetric:
            self._domain.require_same(self._range, "range of a symmetric operator")
        self._verify_properties(matrix)
        self._format = DenseFormat(matrix, bkd)

    def _verify_properties(self, matrix: Array) -> None:
        if matrix.shape[0] == 0:
            return
        bkd, props = self._bkd, self._properties
        # Checks only: converting the reductions to floats detaches
        # nothing that is returned.
        if props.symmetric:
            asymmetry = bkd.to_float(bkd.max(bkd.abs(matrix - matrix.T)))
            scale = bkd.to_float(bkd.max(bkd.abs(matrix)))
            if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
                raise ValueError(
                    f"matrix declared symmetric is not: max |A - A^T| = "
                    f"{asymmetry:.3e} against max |A| = {scale:.3e}"
                )
        if props.diagonal:
            off_diagonal = matrix - bkd.diag(bkd.diag(matrix))
            if bkd.to_float(bkd.max(bkd.abs(off_diagonal))) > 0.0:
                raise ValueError("matrix declared diagonal has off-diagonal nonzeros")
        if props.identity:
            n = matrix.shape[0]
            deviation = matrix - bkd.eye(n, dtype=self._dtype)
            if bkd.to_float(bkd.max(bkd.abs(deviation))) > 0.0:
                raise ValueError(
                    "matrix declared identity has a diagonal entry other than 1"
                )

    @classmethod
    def identity(
        cls, descriptor: SpaceDescriptor, bkd: Backend[Array]
    ) -> "BackendDenseOperator[Array]":
        """The identity on ``descriptor``, stored densely so it combines with
        other dense operators without leaving the format."""
        return cls(
            bkd.eye(descriptor.size, dtype=bkd.default_dtype()),
            descriptor,
            bkd,
            properties=OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True
            ),
        )

    @classmethod
    def from_diagonal(
        cls, d: Array, descriptor: SpaceDescriptor, bkd: Backend[Array]
    ) -> "BackendDenseOperator[Array]":
        """The diagonal operator with diagonal ``d``, a single vector of
        ``descriptor``, stored densely. Declared ``spd`` when ``d`` is
        positive."""
        if tuple(d.shape) != (descriptor.size, 1):
            raise ValueError(
                f"d must be a single vector of shape ({descriptor.size}, 1), "
                f"got {tuple(d.shape)}"
            )
        spd = descriptor.size == 0 or bkd.to_float(bkd.min(d)) > 0.0
        return cls(
            bkd.diag(bkd.reshape(d, (descriptor.size,))),
            descriptor,
            bkd,
            properties=OperatorProperties(symmetric=True, spd=spd, diagonal=True),
        )

    def domain(self) -> SpaceDescriptor:
        return self._domain

    def range(self) -> SpaceDescriptor:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> DenseFormat[Array]:
        return self._format

    @staticmethod
    def _check_apply(x: Array, out: Optional[Array], src: SpaceDescriptor) -> None:
        if x.ndim != 2 or x.shape[0] != src.size:
            raise ValueError(
                f"input must have shape ({src.size}, ncols), got {tuple(x.shape)}"
            )
        if out is x:
            raise ValueError("out must not alias the input of apply")

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(x, out, self._domain)
        return self._bkd.dot(self._format.matrix(), x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        self._check_apply(y, out, self._range)
        return self._bkd.dot(self._format.matrix().T, y)

    def diagonal(self) -> Array:
        self._domain.require_same(
            self._range, "range of an operator asked for its diagonal"
        )
        return self._bkd.reshape(
            self._bkd.diag(self._format.matrix()), (self._domain.size, 1)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self._domain!r}, "
            f"range={self._range!r}, {self._properties!r})"
        )


class _DenseLUHandle(SolverHandleBase[Array]):
    """A backend LU factorization, solving with A or, sharing it, A^T."""

    def __init__(
        self,
        op: ActionOperatorProtocol[Array, Any],
        factors: Optional[tuple[Array, Array]],
        failure: str,
        transposed: bool,
        bkd: Backend[Array],
    ) -> None:
        super().__init__(op)
        self._op = op
        self._factors = factors
        self._failure = failure
        self._transposed = transposed
        self._bkd = bkd

    def solve(
        self, rhs: Array, tol: Optional[float] = None, x0: Optional[Array] = None
    ) -> tuple[Array, SolveInfo]:
        if rhs.ndim != 2 or rhs.shape[0] != self.domain().size:
            raise ValueError(
                f"rhs must have shape ({self.domain().size}, ncols), "
                f"got {tuple(rhs.shape)}"
            )
        if self._factors is None:
            return rhs * float("nan"), SolveInfo(False, message=self._failure)
        lu, pivots = self._factors
        solution = self._bkd.lu_solve(lu, pivots, rhs, adjoint=self._transposed)
        if not _all_finite(solution, self._bkd):
            return solution, SolveInfo(False, message="non-finite solution")
        return solution, SolveInfo(True)

    def transpose(self) -> "_DenseLUHandle[Array]":
        return _DenseLUHandle(
            Transposed(self._op),
            self._factors,
            self._failure,
            not self._transposed,
            self._bkd,
        )


class BackendDenseLU(Generic[Array]):
    """Dense LU through the backend, for operators stored densely.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for :class:`DenseFormat`. Factored once at ``setup``; the backend's
    LU is differentiable, so a torch graph through the matrix and the
    right-hand side is preserved. A singular matrix gives handles whose
    solves report ``converged=False``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[Array, DenseFormat[Array]]
    ) -> SolverHandleProtocol[Array]:
        fmt = require_format(op, DenseFormat, "BackendDenseLU")
        require_square(op, "BackendDenseLU")
        factors: Optional[tuple[Array, Array]] = None
        failure = ""
        try:
            # An exactly singular matrix only warns here; the solve then
            # reports it as non-finite.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                factors = self._bkd.lu_factor(fmt.matrix())
        except (RuntimeError, ValueError) as error:
            failure = f"factorization failed: {error}"
        return _DenseLUHandle(op, factors, failure, False, self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
