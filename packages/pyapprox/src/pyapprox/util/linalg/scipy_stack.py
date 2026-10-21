r"""The NumPy/SciPy linear-algebra stack.

A single-process stack whose vectors are NumPy arrays of shape
``(size, ncols)``: a single vector is ``(size, 1)``, never 1-D. It is a
bound implementation: it works on NumPy arrays only, writes in place
whenever ``out`` is given, and is not differentiable.

The scalar type ``S`` is the stack's configured precision, a type
parameter given once where vectors or matrices are created
(``NumpyVectorSpace(descriptor, np.float32)``); every descriptor the
stack meets must name the same type.
"""

from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import LinearOperator, SuperLU, cg, gmres, splu

from pyapprox.util.linalg.operators.algebra import Transposed
from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor, check_column
from pyapprox.util.linalg.operators.formats import FormatProtocol
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
    merge_column_reports,
    require_format,
    require_square,
)

S = TypeVar("S", np.float32, np.float64, np.complex64, np.complex128)
# A NumPy vector type of the stack, for code that inspects arrays but
# does no arithmetic on them.
H = TypeVar("H", bound=NDArray[np.generic])
# The vector type of a row set given to a row edit; see FormatProtocol.
W = TypeVar("W")
# The format of an operator a Krylov solver applies but does not inspect.
G = TypeVar("G", bound=FormatProtocol)
# A vector block of any of the stack's scalar types, for the Krylov handles,
# which never depend on which: the dtype truly varies.
_Block = NDArray[Any]

_SYMMETRY_RTOL = 1e-12


def require_scalar(descriptor: SpaceDescriptor, scalar: type[S], what: str) -> None:
    """Refuse a space whose dtype is not the configured scalar type."""
    configured = np.dtype(scalar).name
    if descriptor.dtype != configured:
        raise ValueError(
            f"{what}: space {descriptor.name!r} has dtype {descriptor.dtype!r}, "
            f"not the configured scalar type {configured!r}"
        )


def _check_block(
    x: NDArray[np.generic], descriptor: SpaceDescriptor, name: str
) -> None:
    if x.ndim != 2 or x.shape[0] != descriptor.size:
        raise ValueError(
            f"{name} must have shape ({descriptor.size}, ncols), got {x.shape}"
        )
    if x.dtype.name != descriptor.dtype:
        raise ValueError(f"{name} must have dtype {descriptor.dtype}, got {x.dtype}")


def _check_alias(
    out: NDArray[np.generic],
    accumulator: Optional[NDArray[np.generic]],
    others: tuple[NDArray[np.generic], ...],
) -> None:
    """Check ``out``: it may be ``accumulator`` itself, and must share no
    storage with anything else (including ``others`` that happen to be the
    accumulator)."""
    operands = list(others)
    if accumulator is not None and out is not accumulator:
        operands.append(accumulator)
    for operand in operands:
        if np.may_share_memory(out, operand):
            raise ValueError("out shares storage with an operand it may not alias")


class NumpyVectorSpace(Generic[S]):
    """Blocks of vectors stored as NumPy arrays of shape ``(size, ncols)``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.InnerProductSpaceProtocol`
    with ``NDArray[S]`` as both the vector and the reduction type. Every
    operation given ``out`` writes into it and returns it.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The space of the vectors.
    scalar : type
        The configured scalar type, one of ``np.float32``, ``np.float64``,
        ``np.complex64``, ``np.complex128``; the descriptor's dtype must
        name it.

    Raises
    ------
    ValueError
        If the descriptor's dtype is not ``scalar``. From any operation, if
        a vector is not a 2-D array of that dtype with ``descriptor.size``
        rows, if an ``out`` has a different shape from the result, or if
        ``out`` aliases an operand it may not.
    """

    def __init__(self, descriptor: SpaceDescriptor, scalar: type[S]) -> None:
        require_scalar(descriptor, scalar, "NumpyVectorSpace")
        self._descriptor = descriptor
        self._scalar: type[S] = scalar

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def _check(self, x: NDArray[S], name: str) -> None:
        _check_block(x, self._descriptor, name)

    def _check_out(
        self,
        out: NDArray[S],
        ncols: int,
        accumulator: Optional[NDArray[S]],
        others: tuple[NDArray[S], ...],
    ) -> None:
        self._check(out, "out")
        if out.shape[1] != ncols:
            raise ValueError(
                f"out must have shape ({self._descriptor.size}, {ncols}), "
                f"got {out.shape}"
            )
        _check_alias(out, accumulator, others)

    def zeros(self, ncols: int = 1) -> NDArray[S]:
        return np.zeros((self._descriptor.size, ncols), dtype=self._scalar)

    def zeros_like(self, x: NDArray[S]) -> NDArray[S]:
        self._check(x, "x")
        return np.zeros_like(x)

    def ncols(self, x: NDArray[S]) -> int:
        self._check(x, "x")
        return int(x.shape[1])

    def column(self, x: NDArray[S], k: int) -> NDArray[S]:
        self._check(x, "x")
        check_column(k, x.shape[1])
        return x[:, k : k + 1].copy()

    def concatenate(self, blocks: Sequence[NDArray[S]]) -> NDArray[S]:
        for block in blocks:
            self._check(block, "block")
        return np.hstack(blocks)

    def copy(self, x: NDArray[S], out: Optional[NDArray[S]] = None) -> NDArray[S]:
        self._check(x, "x")
        if out is None:
            return x.copy()
        self._check_out(out, x.shape[1], None, (x,))
        np.copyto(out, x)
        return out

    def scale(
        self, a: float, x: NDArray[S], out: Optional[NDArray[S]] = None
    ) -> NDArray[S]:
        self._check(x, "x")
        if out is None:
            return a * x
        self._check_out(out, x.shape[1], x, ())
        np.multiply(x, a, out=out)
        return out

    def axpby(
        self,
        a: float,
        x: NDArray[S],
        b: float,
        y: NDArray[S],
        out: Optional[NDArray[S]] = None,
    ) -> NDArray[S]:
        self._check(x, "x")
        self._check(y, "y")
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )
        if out is None:
            return a * x + b * y
        self._check_out(out, x.shape[1], y, (x,))
        # out is y or disjoint from both, so scaling y first is safe.
        np.multiply(y, b, out=out)
        out += a * x
        return out

    def dot(self, x: NDArray[S], y: NDArray[S]) -> NDArray[S]:
        self._check(x, "x")
        self._check(y, "y")
        return np.conj(x).T @ y

    def norm(self, x: NDArray[S]) -> NDArray[S]:
        """The norms in the space's scalar type (real-valued for a
        complex space)."""
        self._check(x, "x")
        # numpy's stubs type a norm along an axis as Any.
        norm: NDArray[S] = np.linalg.norm(x, axis=0, keepdims=True).astype(self._scalar)
        return norm

    def divide_entries(
        self, x: NDArray[S], d: NDArray[S], out: Optional[NDArray[S]] = None
    ) -> NDArray[S]:
        self._check(x, "x")
        self._check(d, "d")
        if d.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {d.shape}")
        if out is None:
            return x / d
        self._check_out(out, x.shape[1], x, (d,))
        np.divide(x, d, out=out)
        return out

    def all_finite(self, x: NDArray[S]) -> bool:
        self._check(x, "x")
        return bool(np.all(np.isfinite(x)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._descriptor!r})"


class NumpyRowSet:
    """A set of rows of a space, stored as a sorted integer index array.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.RowSetProtocol` for
    NumPy blocks of shape ``(size, ncols)``. Every operation given ``out``
    writes into it and returns it.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The space whose rows are selected.
    rows : sequence of int, or ndarray of int64
        Row indices; duplicates are removed and order is ignored.

    Raises
    ------
    ValueError
        If ``rows`` is not one-dimensional or an index is outside
        ``[0, descriptor.size)``.
    """

    def __init__(
        self,
        descriptor: SpaceDescriptor,
        rows: Union[Sequence[int], NDArray[np.int64]],
    ) -> None:
        indices = np.unique(np.asarray(rows, dtype=np.intp))
        if indices.ndim != 1:
            raise ValueError(f"rows must be one-dimensional, got shape {indices.shape}")
        if indices.size and (indices[0] < 0 or indices[-1] >= descriptor.size):
            raise ValueError(
                f"row indices must lie in [0, {descriptor.size}), got "
                f"[{indices[0]}, {indices[-1]}]"
            )
        indices.setflags(write=False)
        self._descriptor = descriptor
        self._indices = indices

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def count(self) -> int:
        return int(self._indices.size)

    def indices(self) -> NDArray[np.intp]:
        """The sorted row indices (read-only), for this stack's formats."""
        return self._indices

    def _start(
        self, y: NDArray[S], out: Optional[NDArray[S]], others: tuple[NDArray[S], ...]
    ) -> NDArray[S]:
        """Check operands and return the buffer holding a copy of ``y``."""
        _check_block(y, self._descriptor, "y")
        for other in others:
            _check_block(other, self._descriptor, "x")
            if other.shape != y.shape:
                raise ValueError(
                    f"x and y must have the same shape, got {other.shape} and {y.shape}"
                )
        if out is None:
            return y.copy()
        _check_block(out, self._descriptor, "out")
        if out.shape != y.shape:
            raise ValueError(f"out must have shape {y.shape}, got {out.shape}")
        _check_alias(out, y, others)
        if out is not y:
            np.copyto(out, y)
        return out

    def zero_rows(self, x: NDArray[S], out: Optional[NDArray[S]] = None) -> NDArray[S]:
        result = self._start(x, out, ())
        result[self._indices] = 0.0
        return result

    def replace_rows(
        self, y: NDArray[S], x: NDArray[S], out: Optional[NDArray[S]] = None
    ) -> NDArray[S]:
        result = self._start(y, out, (x,))
        result[self._indices] = x[self._indices]
        return result

    def add_rows(
        self, y: NDArray[S], x: NDArray[S], out: Optional[NDArray[S]] = None
    ) -> NDArray[S]:
        result = self._start(y, out, (x,))
        result[self._indices] += x[self._indices]
        return result

    def difference(self, other: "NumpyRowSet") -> "NumpyRowSet":
        self._descriptor.require_same(other.descriptor(), "row set to subtract")
        remaining = np.setdiff1d(self._indices, other.indices(), assume_unique=True)
        return NumpyRowSet(self._descriptor, [int(row) for row in remaining])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._descriptor!r}, count={self.count()})"


def _numpy_row_indices(rows: RowSetProtocol[W]) -> NDArray[np.intp]:
    if not isinstance(rows, NumpyRowSet):
        raise TypeError(
            f"CSRFormat row edits need a NumpyRowSet, got {type(rows).__name__}"
        )
    return rows.indices()


def _csr_zero_rows(matrix: csr_matrix, indices: NDArray[np.intp]) -> csr_matrix:
    """A copy of ``matrix`` with the rows ``indices`` structurally empty.

    The row of each stored entry is expanded from ``indptr``, so the cost
    is O(nnz) with no per-row Python loop.
    """
    result = matrix.copy()
    in_set = np.zeros(result.shape[0], dtype=bool)
    in_set[indices] = True
    entry_rows = np.repeat(np.arange(result.shape[0]), np.diff(result.indptr))
    result.data[in_set[entry_rows]] = 0.0
    result.eliminate_zeros()
    return result


class CSRFormat:
    """Compressed sparse row storage, the format of :class:`ScipyCSROperator`.

    A direct solver that factors a CSR matrix accepts operators of this
    format and reads the matrix through :meth:`matrix`.

    Parameters
    ----------
    matrix : csr_matrix
        The stored matrix.
    """

    def __init__(self, matrix: csr_matrix) -> None:
        self._matrix = matrix

    def matrix(self) -> csr_matrix:
        """The stored matrix. Not to be modified: operators are immutable."""
        return self._matrix

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, "CSRFormat"]]
    ) -> "CSRFormat":
        total = coefficient * self._matrix
        for c, other in others:
            total = total + c * other._matrix
        return CSRFormat(total.tocsr())

    def transposed(self) -> "CSRFormat":
        return CSRFormat(self._matrix.T.tocsr())

    def rows_replaced(self, rows: RowSetProtocol[W]) -> "CSRFormat":
        indices = _numpy_row_indices(rows)
        emptied = _csr_zero_rows(self._matrix, indices)
        # The rows are now structurally empty, so adding the identity
        # restricted to them sets each diagonal to 1 whether or not it was
        # in the original pattern, and leaves every other row unchanged.
        ones = np.ones(indices.shape[0], dtype=emptied.dtype)
        identity_rows = csr_matrix((ones, (indices, indices)), shape=emptied.shape)
        return CSRFormat((emptied + identity_rows).tocsr())

    def rows_zeroed(self, rows: RowSetProtocol[W]) -> "CSRFormat":
        return CSRFormat(_csr_zero_rows(self._matrix, _numpy_row_indices(rows)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._matrix.shape})"


class ScipyCSROperator(Generic[S]):
    """A linear operator stored as a SciPy CSR matrix.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.protocols.DiagonalCapableProtocol`
    with ``NDArray[S]`` as vectors and :class:`CSRFormat` as its format.
    ``apply`` and ``apply_transpose`` copy into ``out`` when it is given.

    Parameters
    ----------
    matrix : scipy sparse matrix
        Shape ``(range.size, domain.size)``, of dtype ``scalar``. Stored as
        CSR.
    domain : SpaceDescriptor
        The space of the input of ``apply``.
    range : SpaceDescriptor, optional
        The space of the output of ``apply``. Defaults to ``domain``.
    properties : OperatorProperties, optional
        Declared properties. ``symmetric``, ``diagonal`` and ``identity``
        are verified, in time proportional to the number of nonzeros;
        ``spd`` is not (that would cost a factorization).
    scalar : type
        The configured scalar type; both spaces' dtypes must name it.

    Raises
    ------
    TypeError
        If ``matrix`` is not a SciPy sparse matrix.
    ValueError
        If the shape or dtype disagrees with the descriptors or ``scalar``,
        if a symmetric operator's domain and range differ, or if a
        verified property does not hold.
    """

    def __init__(
        self,
        matrix: spmatrix,
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
        *,
        scalar: type[S],
    ) -> None:
        if not issparse(matrix):
            raise TypeError(
                "ScipyCSROperator requires a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        self._domain = domain
        self._range = domain if range is None else range
        self._properties = OperatorProperties() if properties is None else properties
        require_scalar(self._domain, scalar, "ScipyCSROperator")
        require_scalar(self._range, scalar, "ScipyCSROperator")
        csr = matrix.tocsr()
        if csr.dtype != np.dtype(scalar):
            raise ValueError(
                f"matrix dtype {csr.dtype} differs from the configured scalar "
                f"type {np.dtype(scalar).name}"
            )
        _check_operator_layouts(
            tuple(csr.shape), self._domain, self._range, self._properties
        )
        _verify_csr_properties(csr, self._properties)
        self._format = CSRFormat(csr)

    @classmethod
    def identity(
        cls, descriptor: SpaceDescriptor, scalar: type[S]
    ) -> "ScipyCSROperator[S]":
        """The identity on ``descriptor``, stored as CSR so it combines with
        other CSR operators without leaving the format."""
        matrix = sparse_identity(descriptor.size, dtype=scalar, format="csr")
        return cls(
            matrix,
            descriptor,
            properties=OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True
            ),
            scalar=scalar,
        )

    @classmethod
    def from_diagonal(
        cls, d: NDArray[S], descriptor: SpaceDescriptor
    ) -> "ScipyCSROperator[S]":
        """The diagonal operator with diagonal ``d``, a single vector of
        ``descriptor``, stored as CSR; the scalar type is ``d``'s. Declared
        ``spd`` when ``d`` is real and positive."""
        _check_block(d, descriptor, "d")
        if d.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {d.shape}")
        spd = bool(np.isrealobj(d) and np.all(d.real > 0))
        return cls(
            diags(d[:, 0], format="csr"),
            descriptor,
            properties=OperatorProperties(symmetric=True, spd=spd, diagonal=True),
            scalar=d.dtype.type,
        )

    def domain(self) -> SpaceDescriptor:
        return self._domain

    def range(self) -> SpaceDescriptor:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> CSRFormat:
        return self._format

    def _into(self, result: NDArray[S], out: Optional[NDArray[S]]) -> NDArray[S]:
        if out is None:
            return result
        np.copyto(out, result)
        return out

    def _check_apply(
        self,
        x: NDArray[S],
        out: Optional[NDArray[S]],
        src: SpaceDescriptor,
        dst: SpaceDescriptor,
    ) -> None:
        _check_block(x, src, "input")
        if out is None:
            return
        _check_block(out, dst, "out")
        if out.shape[1] != x.shape[1]:
            raise ValueError(
                f"out must have {x.shape[1]} columns, got shape {out.shape}"
            )
        if np.may_share_memory(out, x):
            raise ValueError("out must not alias the input of apply")

    def apply(self, x: NDArray[S], out: Optional[NDArray[S]] = None) -> NDArray[S]:
        self._check_apply(x, out, self._domain, self._range)
        # scipy's sparse product is untyped; the result has x's scalar type.
        result: NDArray[S] = self._format.matrix() @ x
        return self._into(result, out)

    def apply_transpose(
        self, y: NDArray[S], out: Optional[NDArray[S]] = None
    ) -> NDArray[S]:
        self._check_apply(y, out, self._range, self._domain)
        result: NDArray[S] = self._format.matrix().T @ y
        return self._into(result, out)

    def diagonal(self) -> NDArray[S]:
        self._domain.require_same(
            self._range, "range of an operator asked for its diagonal"
        )
        diagonal: NDArray[S] = self._format.matrix().diagonal()
        return diagonal.reshape(-1, 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self._domain!r}, "
            f"range={self._range!r}, {self._properties!r})"
        )


class _SparseLUHandle(SolverHandleBase[H]):
    """A SuperLU factorization, solving with A or, sharing it, A^T."""

    def __init__(
        self,
        op: ActionOperatorProtocol[H, CSRFormat],
        factor: Optional[SuperLU],
        failure: str,
        transposed: bool,
    ) -> None:
        super().__init__(op)
        self._op: ActionOperatorProtocol[H, CSRFormat] = op
        self._factor = factor
        self._failure = failure
        self._transposed = transposed

    def solve(
        self, rhs: H, tol: Optional[float] = None, x0: Optional[H] = None
    ) -> tuple[H, SolveInfo]:
        _check_block(rhs, self.domain(), "rhs")
        if self._factor is None:
            failed: H = np.full_like(rhs, np.nan)
            return failed, SolveInfo(False, message=self._failure)
        # SuperLU is untyped; the solution has the right-hand side's type.
        solution: H = self._factor.solve(rhs, trans="T" if self._transposed else "N")
        if not np.all(np.isfinite(solution)):
            return solution, SolveInfo(False, message="non-finite solution")
        return solution, SolveInfo(True)

    def transpose(self) -> "_SparseLUHandle[H]":
        return _SparseLUHandle(
            Transposed(self._op), self._factor, self._failure, not self._transposed
        )


class ScipySparseLU:
    """Sparse LU (SuperLU) for operators stored as CSR.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for :class:`CSRFormat`. One factorization serves both the solve and
    the transposed solve. A singular matrix gives handles whose solves
    report ``converged=False``; ``setup`` does not raise for it.
    """

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[NDArray[S], CSRFormat]
    ) -> SolverHandleProtocol[NDArray[S]]:
        fmt = require_format(op, CSRFormat, "ScipySparseLU")
        require_square(op, "ScipySparseLU")
        try:
            factor: Optional[SuperLU] = splu(fmt.matrix().tocsc())
            failure = ""
        except RuntimeError as error:
            factor, failure = None, f"factorization failed: {error}"
        return _SparseLUHandle(op, factor, failure, transposed=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _check_operator_layouts(
    shape: tuple[int, ...],
    domain: SpaceDescriptor,
    range_: SpaceDescriptor,
    properties: OperatorProperties,
) -> None:
    if shape != (range_.size, domain.size):
        raise ValueError(
            f"matrix must have shape (range.size, domain.size) = "
            f"({range_.size}, {domain.size}), got {shape}"
        )
    if properties.symmetric:
        domain.require_same(range_, "range of a symmetric operator")


def _verify_csr_properties(csr: csr_matrix, properties: OperatorProperties) -> None:
    if csr.nnz == 0:
        if properties.identity and csr.shape[0] > 0:
            raise ValueError("matrix declared identity is zero")
        return
    if properties.symmetric:
        # Sparse A - A^T stays sparse: an O(nnz) check, never densified.
        asymmetry = float(abs(csr - csr.T).max())
        scale = float(abs(csr).max())
        if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
            raise ValueError(
                f"matrix declared symmetric is not: max |A - A^T| = "
                f"{asymmetry:.3e} against max |A| = {scale:.3e}"
            )
    if properties.diagonal:
        off_diagonal = csr - diags(csr.diagonal(), format="csr")
        if off_diagonal.count_nonzero() > 0:
            raise ValueError("matrix declared diagonal has off-diagonal nonzeros")
    if properties.identity and not np.all(csr.diagonal() == 1):
        raise ValueError("matrix declared identity has a diagonal entry other than 1")


class _ScipyKrylovHandle(SolverHandleBase[_Block], Generic[G]):
    """What the SciPy Krylov handles share: the operator as a SciPy
    ``LinearOperator``, the checks, the true-residual test and the
    report. A subclass runs its SciPy method."""

    def __init__(
        self, op: ActionOperatorProtocol[_Block, G], tol: float, maxiter: int
    ) -> None:
        super().__init__(op)
        self._op: ActionOperatorProtocol[_Block, G] = op
        self._tol = tol
        self._maxiter = maxiter
        size = op.domain().size
        self._linear: LinearOperator = LinearOperator(
            (size, size), matvec=self._matvec, dtype=np.dtype(op.domain().dtype)
        )

    def _matvec(self, v: _Block) -> _Block:
        return self._op.apply(v.reshape(-1, 1))[:, 0]

    def solve(
        self,
        rhs: _Block,
        tol: Optional[float] = None,
        x0: Optional[_Block] = None,
    ) -> tuple[_Block, SolveInfo]:
        r"""Solve :math:`A x = b`, each column to ``tol`` relative to its
        own :math:`\|b\|`.

        The columns are solved one after another, each from its column
        of ``x0``; the report is merged by :func:`merge_column_reports`.

        Raises
        ------
        ValueError
            If ``rhs`` or ``x0`` is not a block of the space, or ``x0``
            has a different number of columns from ``rhs``.
        """
        _check_block(rhs, self.domain(), "rhs")
        ncols = rhs.shape[1]
        if x0 is not None:
            _check_block(x0, self.range(), "x0")
            if x0.shape[1] != ncols:
                raise ValueError(
                    f"x0 must have {ncols} columns like rhs, got {x0.shape[1]}"
                )
        if ncols == 1:
            return self._solve_vector(rhs, tol, x0)
        results = [
            self._solve_vector(
                rhs[:, k : k + 1], tol, None if x0 is None else x0[:, k : k + 1]
            )
            for k in range(ncols)
        ]
        solution = (
            np.hstack([x for x, _ in results]) if results else np.zeros_like(rhs)
        )
        return solution, merge_column_reports([info for _, info in results])

    def _solve_vector(
        self, rhs: _Block, tol: Optional[float], x0: Optional[_Block]
    ) -> tuple[_Block, SolveInfo]:
        """Solve for a single right-hand side."""
        tolerance = self._tol if tol is None else tol
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return np.zeros_like(rhs), SolveInfo(True, 0, 0.0)
        counter = [0]
        start = None if x0 is None else x0[:, 0]
        x = self._run(rhs[:, 0], start, tolerance, counter).reshape(-1, 1)
        residual_norm = float(np.linalg.norm(rhs - self._op.apply(x)))
        if not np.all(np.isfinite(x)) or not np.isfinite(residual_norm):
            return x, SolveInfo(False, counter[0], message="non-finite iterate")
        if residual_norm > tolerance * rhs_norm:
            return x, SolveInfo(
                False,
                counter[0],
                residual_norm,
                f"relative residual {residual_norm / rhs_norm:.3e} above "
                f"{tolerance:.1e} after {counter[0]} iterations",
            )
        return x, SolveInfo(True, counter[0], residual_norm)

    @abstractmethod
    def _run(
        self,
        b: _Block,
        x0: Optional[_Block],
        tol: float,
        counter: list[int],
    ) -> _Block:
        """Run the SciPy method, counting iterations in ``counter[0]``."""
        ...


class _ScipyCGHandle(_ScipyKrylovHandle[G]):
    def _run(
        self,
        b: _Block,
        x0: Optional[_Block],
        tol: float,
        counter: list[int],
    ) -> _Block:
        def count(_: _Block) -> None:
            counter[0] += 1

        solution: _Block = cg(
            self._linear,
            b,
            x0=x0,
            rtol=tol,
            atol=0.0,
            maxiter=self._maxiter,
            callback=count,
        )[0]
        return solution

    def transpose(self) -> "_ScipyCGHandle[G]":
        return _ScipyCGHandle(Transposed(self._op), self._tol, self._maxiter)


class _ScipyGMRESHandle(_ScipyKrylovHandle[G]):
    def __init__(
        self,
        op: ActionOperatorProtocol[_Block, G],
        tol: float,
        maxiter: int,
        restart: int,
    ) -> None:
        super().__init__(op, tol, maxiter)
        self._restart = restart

    def _run(
        self,
        b: _Block,
        x0: Optional[_Block],
        tol: float,
        counter: list[int],
    ) -> _Block:
        def count(_: float) -> None:
            counter[0] += 1

        # SciPy counts maxiter in restart cycles and cannot stop inside
        # one; ours counts operator applications, so cycles are shortened
        # to the limit and the limit rounds up to whole cycles.
        restart = min(self._restart, self._maxiter)
        cycles = -(-self._maxiter // restart)
        solution: _Block = gmres(
            self._linear,
            b,
            x0=x0,
            rtol=tol,
            atol=0.0,
            restart=restart,
            maxiter=cycles,
            callback=count,
            callback_type="pr_norm",
        )[0]
        return solution

    def transpose(self) -> "_ScipyGMRESHandle[G]":
        return _ScipyGMRESHandle(
            Transposed(self._op), self._tol, self._maxiter, self._restart
        )


def _check_krylov_settings(tol: float, maxiter: int) -> None:
    if tol <= 0.0 or maxiter < 1:
        raise ValueError(f"need tol > 0 and maxiter >= 1, got {tol}, {maxiter}")


class ScipyCG:
    """SciPy's conjugate gradients, for the NumPy/SciPy stack.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for every format: the operator reaches SciPy as a ``LinearOperator``
    applying it, never as a matrix, so stored and matrix-free operators
    are treated alike. Behaves as
    :class:`~pyapprox.util.linalg.operators.ConjugateGradient`
    (true-residual test, single right-hand sides, no symmetry check) with
    SciPy's implementation of the iterations.

    Parameters
    ----------
    tol : float, default 1e-10
        Relative residual to reach.
    maxiter : int, default 1000
        Iteration limit.
    """

    def __init__(self, tol: float = 1e-10, maxiter: int = 1000) -> None:
        _check_krylov_settings(tol, maxiter)
        self._tol = tol
        self._maxiter = maxiter

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[NDArray[S], G]
    ) -> SolverHandleProtocol[NDArray[S]]:
        require_square(op, "ScipyCG")
        op.domain().require_same(op.range(), "range of the operator given to ScipyCG")
        return _ScipyCGHandle(op, self._tol, self._maxiter)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tol={self._tol}, maxiter={self._maxiter})"


class ScipyGMRES:
    """SciPy's restarted GMRES, for the NumPy/SciPy stack.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for every format, as :class:`ScipyCG` does. Behaves as
    :class:`~pyapprox.util.linalg.operators.GMRES` (true-residual test,
    single right-hand sides, ``maxiter`` counting operator applications)
    with SciPy's implementation of the iterations.

    Parameters
    ----------
    tol : float, default 1e-10
        Relative residual to reach.
    maxiter : int, default 1000
        Limit on operator applications, over all cycles, rounded up to a
        whole number of cycles (SciPy cannot stop inside one).
    restart : int, default 50
        Basis size per cycle; at most ``maxiter``.
    """

    def __init__(
        self, tol: float = 1e-10, maxiter: int = 1000, restart: int = 50
    ) -> None:
        _check_krylov_settings(tol, maxiter)
        if restart < 1:
            raise ValueError(f"need restart >= 1, got {restart}")
        self._tol = tol
        self._maxiter = maxiter
        self._restart = restart

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[NDArray[S], G]
    ) -> SolverHandleProtocol[NDArray[S]]:
        require_square(op, "ScipyGMRES")
        op.domain().require_same(
            op.range(), "range of the operator given to ScipyGMRES"
        )
        return _ScipyGMRESHandle(op, self._tol, self._maxiter, self._restart)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tol={self._tol}, maxiter={self._maxiter}, "
            f"restart={self._restart})"
        )
