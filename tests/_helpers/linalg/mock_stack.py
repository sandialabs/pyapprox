"""A mock linear-algebra stack whose vectors cannot be looked inside.

Proves that code written against the operator protocols makes no
assumption about how a stack stores its vectors and matrices. Vectors
are :class:`MockVector` handles: indexing, ``@``, arithmetic and
conversion to a NumPy array all raise, and every handle carries its
space descriptor, so each operation checks that its operands belong to
the right space (a check a plain array stack cannot make). Operators
store coordinate triplets in :class:`MockFormat`, a format no other
stack's solver accepts, and :class:`MockLU` is the only solver that
accepts it. :class:`MockAssemblyStack` builds into that format and
converts bulk sparse input into it.

Internally everything is NumPy; only this module looks inside a handle,
and tests read one only through :func:`peek`.
"""

import warnings
from typing import Optional, Sequence, TypeVar, Union

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.sparse import issparse, spmatrix

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    OperatorProperties,
    RowSetProtocol,
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
    SpaceDescriptor,
    Transposed,
)
from pyapprox.util.linalg.operators.descriptor import check_column
from pyapprox.util.linalg.operators.solvers import require_format, require_square

# The mock holds float64 entries, indexed by intp row and column arrays.
_Array = NDArray[np.float64]
_Index = NDArray[np.intp]
# Element DOF maps, as the assembly protocols type them.
_Dofs = NDArray[np.int64]
# The vector type of a row set given to a row edit; see FormatProtocol.
W = TypeVar("W")
_RTOL = 1e-12


class MockVector:
    """An opaque block of vectors of one space."""

    __slots__ = ("_data", "_descriptor")

    def __init__(self, data: _Array, descriptor: SpaceDescriptor) -> None:
        if data.ndim != 2 or data.shape[0] != descriptor.size:
            raise ValueError(
                f"data must have shape ({descriptor.size}, ncols), got {data.shape}"
            )
        self._data = data
        self._descriptor = descriptor

    def __array__(
        self, dtype: Optional[np.dtype[np.float64]] = None, copy: Optional[bool] = None
    ) -> _Array:
        """Refuse conversion: the signature is NumPy's array protocol."""
        raise TypeError("a MockVector is opaque: use the stack's operations")

    def __repr__(self) -> str:
        return f"MockVector({self._descriptor.name!r}, ncols={self._data.shape[1]})"


def peek(x: MockVector) -> _Array:
    """A copy of the entries of ``x``, for test assertions only."""
    return x._data.copy()


def wrap(array: _Array, descriptor: SpaceDescriptor) -> MockVector:
    """A mock vector holding a copy of ``array``, for test setup only."""
    return MockVector(np.array(array, dtype=np.dtype(descriptor.dtype)), descriptor)


def _entries(x: MockVector, descriptor: SpaceDescriptor, name: str) -> _Array:
    """The entries of ``x``, after checking it belongs to ``descriptor``."""
    if not isinstance(x, MockVector):
        raise TypeError(f"{name} must be a MockVector, got {type(x).__name__}")
    descriptor.require_same(x._descriptor, name)
    return x._data


def _write(
    out: Optional[MockVector], result: _Array, descriptor: SpaceDescriptor
) -> MockVector:
    """Write ``result`` into ``out`` in place when given, else wrap it."""
    if out is None:
        return MockVector(result, descriptor)
    target = _entries(out, descriptor, "out")
    if target.shape != result.shape:
        raise ValueError(f"out must have shape {result.shape}, got {target.shape}")
    target[...] = result
    return out


def _refuse_alias(out: Optional[MockVector], *others: MockVector) -> None:
    if out is not None and any(other is out for other in others):
        raise ValueError("out is an operand it may not alias")


class MockVectorSpace:
    """Operations on :class:`MockVector` blocks of one space; writes into
    ``out`` when given."""

    def __init__(self, descriptor: SpaceDescriptor) -> None:
        self._descriptor = descriptor

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def _get(self, x: MockVector, name: str) -> _Array:
        return _entries(x, self._descriptor, name)

    def zeros(self, ncols: int = 1) -> MockVector:
        return MockVector(
            np.zeros((self._descriptor.size, ncols), np.dtype(self._descriptor.dtype)),
            self._descriptor,
        )

    def zeros_like(self, x: MockVector) -> MockVector:
        return self.zeros(self._get(x, "x").shape[1])

    def ncols(self, x: MockVector) -> int:
        return int(self._get(x, "x").shape[1])

    def column(self, x: MockVector, k: int) -> MockVector:
        entries = self._get(x, "x")
        check_column(k, entries.shape[1])
        return MockVector(entries[:, k : k + 1].copy(), self._descriptor)

    def concatenate(self, blocks: Sequence[MockVector]) -> MockVector:
        return MockVector(
            np.hstack([self._get(block, "block") for block in blocks]),
            self._descriptor,
        )

    def copy(self, x: MockVector, out: Optional[MockVector] = None) -> MockVector:
        _refuse_alias(out, x)
        return _write(out, self._get(x, "x").copy(), self._descriptor)

    def scale(
        self, a: float, x: MockVector, out: Optional[MockVector] = None
    ) -> MockVector:
        return _write(out, a * self._get(x, "x"), self._descriptor)

    def axpby(
        self,
        a: float,
        x: MockVector,
        b: float,
        y: MockVector,
        out: Optional[MockVector] = None,
    ) -> MockVector:
        _refuse_alias(out, x)
        xs, ys = self._get(x, "x"), self._get(y, "y")
        if xs.shape != ys.shape:
            raise ValueError(
                f"x and y must have the same shape, got {xs.shape} and {ys.shape}"
            )
        return _write(out, a * xs + b * ys, self._descriptor)

    def dot(self, x: MockVector, y: MockVector) -> _Array:
        result: _Array = np.conj(self._get(x, "x")).T @ self._get(y, "y")
        return result

    def norm(self, x: MockVector) -> _Array:
        result: _Array = np.linalg.norm(self._get(x, "x"), axis=0, keepdims=True)
        return result

    def divide_entries(
        self, x: MockVector, d: MockVector, out: Optional[MockVector] = None
    ) -> MockVector:
        _refuse_alias(out, d)
        ds = self._get(d, "d")
        if ds.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {ds.shape}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return _write(out, self._get(x, "x") / ds, self._descriptor)

    def all_finite(self, x: MockVector) -> bool:
        return bool(np.all(np.isfinite(self._get(x, "x"))))


class MockRowSet:
    """A set of rows of one space, applied to :class:`MockVector` blocks."""

    def __init__(
        self, descriptor: SpaceDescriptor, rows: Union[Sequence[int], _Dofs]
    ) -> None:
        indices = np.unique(np.asarray(rows, dtype=np.intp))
        if indices.size and (indices[0] < 0 or indices[-1] >= descriptor.size):
            raise ValueError(
                f"row indices must lie in [0, {descriptor.size}), got "
                f"[{indices[0]}, {indices[-1]}]"
            )
        self._descriptor = descriptor
        self._indices = indices

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def count(self) -> int:
        return int(self._indices.size)

    def indices(self) -> NDArray[np.intp]:
        """The sorted row indices, for :class:`MockFormat`."""
        return self._indices

    def _pair(self, y: MockVector, x: MockVector) -> tuple[_Array, _Array]:
        ys = _entries(y, self._descriptor, "y")
        xs = _entries(x, self._descriptor, "x")
        if xs.shape != ys.shape:
            raise ValueError(
                f"x and y must have the same shape, got {xs.shape} and {ys.shape}"
            )
        return ys, xs

    def zero_rows(self, x: MockVector, out: Optional[MockVector] = None) -> MockVector:
        result = _entries(x, self._descriptor, "x").copy()
        result[self._indices] = 0.0
        return _write(out, result, self._descriptor)

    def replace_rows(
        self, y: MockVector, x: MockVector, out: Optional[MockVector] = None
    ) -> MockVector:
        _refuse_alias(out, x)
        ys, xs = self._pair(y, x)
        result = ys.copy()
        result[self._indices] = xs[self._indices]
        return _write(out, result, self._descriptor)

    def add_rows(
        self, y: MockVector, x: MockVector, out: Optional[MockVector] = None
    ) -> MockVector:
        _refuse_alias(out, x)
        ys, xs = self._pair(y, x)
        result = ys.copy()
        result[self._indices] += xs[self._indices]
        return _write(out, result, self._descriptor)

    def difference(self, other: "MockRowSet") -> "MockRowSet":
        self._descriptor.require_same(other.descriptor(), "row set to subtract")
        return MockRowSet(
            self._descriptor,
            list(np.setdiff1d(self._indices, other.indices(), assume_unique=True)),
        )


def _coalesce(
    rows: _Index, cols: _Index, vals: _Array, shape: tuple[int, int]
) -> tuple[_Index, _Index, _Array]:
    """Sum duplicate coordinates; keep explicit zeros out."""
    keys = rows.astype(np.intp) * shape[1] + cols.astype(np.intp)
    unique, inverse = np.unique(keys, return_inverse=True)
    summed = np.zeros(unique.shape[0], dtype=vals.dtype)
    np.add.at(summed, inverse, vals)
    keep = summed != 0.0
    unique, summed = unique[keep], summed[keep]
    return unique // shape[1], unique % shape[1], summed


class MockFormat:
    """Coordinate triplets: the format of :class:`MockOperator`."""

    def __init__(
        self, rows: _Index, cols: _Index, vals: _Array, shape: tuple[int, int]
    ) -> None:
        self._rows, self._cols, self._vals = _coalesce(rows, cols, vals, shape)
        self._shape = shape

    def triplets(self) -> tuple[_Index, _Index, _Array]:
        """The stored entries, for :class:`MockLU`."""
        return self._rows, self._cols, self._vals

    def shape(self) -> tuple[int, int]:
        return self._shape

    def dense(self) -> _Array:
        """The stored matrix densified, for :class:`MockLU` and tests."""
        matrix = np.zeros(self._shape)
        matrix[self._rows, self._cols] = self._vals
        return matrix

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, "MockFormat"]]
    ) -> "MockFormat":
        parts = [(coefficient, self), *others]
        return MockFormat(
            np.concatenate([f._rows for _, f in parts]),
            np.concatenate([f._cols for _, f in parts]),
            np.concatenate([c * f._vals for c, f in parts]),
            self._shape,
        )

    def transposed(self) -> "MockFormat":
        return MockFormat(
            self._cols, self._rows, self._vals, (self._shape[1], self._shape[0])
        )

    @staticmethod
    def _mock_rows(rows: RowSetProtocol[W]) -> "MockRowSet":
        if not isinstance(rows, MockRowSet):
            raise TypeError(
                f"MockFormat row edits need a MockRowSet, got {type(rows).__name__}"
            )
        return rows

    def rows_replaced(self, rows: RowSetProtocol[W]) -> "MockFormat":
        indices = self._mock_rows(rows).indices()
        keep = ~np.isin(self._rows, indices)
        return MockFormat(
            np.concatenate([self._rows[keep], indices]),
            np.concatenate([self._cols[keep], indices]),
            np.concatenate([self._vals[keep], np.ones(indices.shape[0])]),
            self._shape,
        )

    def rows_zeroed(self, rows: RowSetProtocol[W]) -> "MockFormat":
        keep = ~np.isin(self._rows, self._mock_rows(rows).indices())
        return MockFormat(
            self._rows[keep], self._cols[keep], self._vals[keep], self._shape
        )


class MockOperator:
    """An operator stored as :class:`MockFormat`, acting on mock vectors.

    Declared ``symmetric``, ``diagonal`` and ``identity`` are verified.
    """

    def __init__(
        self,
        fmt: MockFormat,
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> None:
        self._domain = domain
        self._range = domain if range is None else range
        self._properties = OperatorProperties() if properties is None else properties
        if fmt.shape() != (self._range.size, self._domain.size):
            raise ValueError(
                f"matrix must have shape (range.size, domain.size) = "
                f"({self._range.size}, {self._domain.size}), got {fmt.shape()}"
            )
        if self._properties.symmetric:
            self._domain.require_same(self._range, "range of a symmetric operator")
        dense = fmt.dense()
        if self._properties.symmetric and not np.allclose(
            dense, dense.T, rtol=_RTOL, atol=0
        ):
            raise ValueError("matrix declared symmetric is not")
        if self._properties.diagonal and np.count_nonzero(
            dense - np.diag(np.diag(dense))
        ):
            raise ValueError("matrix declared diagonal has off-diagonal nonzeros")
        if self._properties.identity and not np.array_equal(
            dense, np.eye(dense.shape[0])
        ):
            raise ValueError(
                "matrix declared identity has a diagonal entry other than 1"
            )
        self._format = fmt

    @classmethod
    def from_dense(
        cls,
        matrix: _Array,
        domain: SpaceDescriptor,
        range: Optional[SpaceDescriptor] = None,
        properties: Optional[OperatorProperties] = None,
    ) -> "MockOperator":
        rows, cols = np.nonzero(matrix)
        fmt = MockFormat(rows, cols, matrix[rows, cols], matrix.shape)
        return cls(fmt, domain, range, properties)

    @classmethod
    def identity(cls, descriptor: SpaceDescriptor) -> "MockOperator":
        n = np.arange(descriptor.size)
        return cls(
            MockFormat(
                n, n, np.ones(descriptor.size), (descriptor.size, descriptor.size)
            ),
            descriptor,
            properties=OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True
            ),
        )

    @classmethod
    def from_diagonal(
        cls, d: MockVector, descriptor: SpaceDescriptor
    ) -> "MockOperator":
        values = _entries(d, descriptor, "d")
        if values.shape[1] != 1:
            raise ValueError(f"d must be a single vector, got shape {values.shape}")
        n = np.arange(descriptor.size)
        spd = bool(np.all(values > 0))
        return cls(
            MockFormat(n, n, values[:, 0], (descriptor.size, descriptor.size)),
            descriptor,
            properties=OperatorProperties(symmetric=True, spd=spd, diagonal=True),
        )

    def domain(self) -> SpaceDescriptor:
        return self._domain

    def range(self) -> SpaceDescriptor:
        return self._range

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MockFormat:
        return self._format

    @staticmethod
    def _product(
        rows: _Index, cols: _Index, vals: _Array, x: _Array, nrows: int
    ) -> _Array:
        result = np.zeros((nrows, x.shape[1]), dtype=x.dtype)
        np.add.at(result, rows, vals[:, None] * x[cols])
        return result

    def apply(self, x: MockVector, out: Optional[MockVector] = None) -> MockVector:
        _refuse_alias(out, x)
        rows, cols, vals = self._format.triplets()
        product = self._product(
            rows, cols, vals, _entries(x, self._domain, "input"), self._range.size
        )
        return _write(out, product, self._range)

    def apply_transpose(
        self, y: MockVector, out: Optional[MockVector] = None
    ) -> MockVector:
        _refuse_alias(out, y)
        rows, cols, vals = self._format.triplets()
        product = self._product(
            cols, rows, vals, _entries(y, self._range, "input"), self._domain.size
        )
        return _write(out, product, self._domain)

    def diagonal(self) -> MockVector:
        self._domain.require_same(
            self._range, "range of an operator asked for its diagonal"
        )
        return MockVector(np.diag(self._format.dense())[:, None].copy(), self._domain)


class _MockLUHandle(SolverHandleBase[MockVector]):
    def __init__(
        self,
        op: ActionOperatorProtocol[MockVector, MockFormat],
        factors: Optional[tuple[_Array, _Index]],
        transposed: bool,
    ) -> None:
        super().__init__(op)
        self._op = op
        self._factors = factors
        self._transposed = transposed

    def solve(
        self,
        rhs: MockVector,
        tol: Optional[float] = None,
        x0: Optional[MockVector] = None,
    ) -> tuple[MockVector, SolveInfo]:
        b = _entries(rhs, self.domain(), "rhs")
        if self._factors is None:
            return MockVector(np.full_like(b, np.nan), self.range()), SolveInfo(
                False, message="factorization failed"
            )
        solution = scipy.linalg.lu_solve(
            self._factors, b, trans=1 if self._transposed else 0
        )
        if not np.all(np.isfinite(solution)):
            return MockVector(solution, self.range()), SolveInfo(
                False, message="non-finite solution"
            )
        return MockVector(solution, self.range()), SolveInfo(True)

    def transpose(self) -> "_MockLUHandle":
        return _MockLUHandle(Transposed(self._op), self._factors, not self._transposed)


class MockLU:
    """Dense LU of a :class:`MockFormat` operator; refuses any other format."""

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[MockVector, MockFormat]
    ) -> SolverHandleProtocol[MockVector]:
        fmt = require_format(op, MockFormat, "MockLU")
        require_square(op, "MockLU")
        factors: Optional[tuple[_Array, _Index]] = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                factors = scipy.linalg.lu_factor(fmt.dense())
        except (ValueError, np.linalg.LinAlgError):
            factors = None
        return _MockLUHandle(op, factors, transposed=False)


def _check_dofs(dofs: _Dofs, size: int, name: str) -> _Index:
    as_int = np.asarray(dofs, dtype=np.intp)
    if as_int.ndim != 2:
        raise ValueError(f"{name} must have shape (nsel, n), got {as_int.shape}")
    if as_int.size and (as_int.min() < 0 or as_int.max() >= size):
        raise ValueError(f"{name} must lie in [0, {size})")
    return as_int


class _MockBulkMatrixSink:
    def __init__(self, sink: "MockMatrixSink") -> None:
        self._sink = sink

    def add_assembled(self, matrix: spmatrix) -> None:
        """Converts the sparse matrix into the mock's own triplets."""
        if not issparse(matrix):
            raise TypeError(
                "add_assembled needs a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        coo = matrix.tocoo()
        self._sink._add(coo.row, coo.col, coo.data)


class MockMatrixSink:
    """Accumulates element blocks into a dictionary of keys, then triplets."""

    def __init__(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
    ) -> None:
        rows = _check_dofs(row_dofs, range.size, "row_dofs")
        cols = _check_dofs(col_dofs, domain.size, "col_dofs")
        keys = (rows[:, :, None] * domain.size + cols[:, None, :]).ravel()
        self._range, self._domain = range, domain
        self._values: dict[int, float] = {int(key): 0.0 for key in keys}

    def _add(self, rows: _Index, cols: _Index, vals: _Array) -> None:
        for key, value in zip(
            (
                np.asarray(rows, np.intp) * self._domain.size
                + np.asarray(cols, np.intp)
            ).ravel(),
            np.asarray(vals).ravel(),
        ):
            if int(key) not in self._values:
                raise ValueError("an entry lies outside the sink's sparsity pattern")
            self._values[int(key)] += float(value)

    def add_element_matrices(
        self,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
        values: _Array,
        weights: Optional[_Array] = None,
    ) -> None:
        rows = _check_dofs(row_dofs, self._range.size, "row_dofs")
        cols = _check_dofs(col_dofs, self._domain.size, "col_dofs")
        scaled = values if weights is None else values * weights[:, None, None]
        self._add(
            np.broadcast_to(rows[:, :, None], scaled.shape),
            np.broadcast_to(cols[:, None, :], scaled.shape),
            scaled,
        )

    def bulk(self) -> _MockBulkMatrixSink:
        return _MockBulkMatrixSink(self)

    def zero(self) -> None:
        for key in self._values:
            self._values[key] = 0.0

    def assemble(self, properties: Optional[OperatorProperties] = None) -> MockOperator:
        keys = np.fromiter(self._values.keys(), dtype=np.intp)
        vals = np.fromiter(self._values.values(), dtype=float)
        fmt = MockFormat(
            keys // self._domain.size,
            keys % self._domain.size,
            vals,
            (self._range.size, self._domain.size),
        )
        return MockOperator(fmt, self._domain, self._range, properties)


class _MockBulkVectorSink:
    def __init__(self, sink: "MockVectorSink") -> None:
        self._sink = sink

    def add_assembled(self, vector: _Array) -> None:
        if vector.shape != self._sink._data.shape:
            raise ValueError(
                f"vector must have shape {self._sink._data.shape}, got {vector.shape}"
            )
        self._sink._data += vector


class MockVectorSink:
    """Accumulates element vectors into a :class:`MockVector`."""

    def __init__(self, descriptor: SpaceDescriptor) -> None:
        self._descriptor = descriptor
        self._data = np.zeros(descriptor.size)

    def add_element_vectors(
        self, row_dofs: _Dofs, values: _Array, weights: Optional[_Array] = None
    ) -> None:
        rows = _check_dofs(row_dofs, self._descriptor.size, "row_dofs")
        scaled = values if weights is None else values * weights[:, None]
        np.add.at(self._data, rows.ravel(), np.asarray(scaled).ravel())

    def bulk(self) -> _MockBulkVectorSink:
        return _MockBulkVectorSink(self)

    def zero(self) -> None:
        self._data[:] = 0.0

    def assemble(self) -> MockVector:
        return MockVector(self._data.copy()[:, None], self._descriptor)


class MockAssemblyStack:
    """Assembly into the mock stack; the prolongation is the identity."""

    def space(self, descriptor: SpaceDescriptor) -> MockVectorSpace:
        return MockVectorSpace(descriptor)

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
    ) -> MockMatrixSink:
        return MockMatrixSink(range, domain, row_dofs, col_dofs)

    def vector_sink(self, descriptor: SpaceDescriptor) -> MockVectorSink:
        return MockVectorSink(descriptor)

    def row_set(self, descriptor: SpaceDescriptor, rows: _Dofs) -> MockRowSet:
        return MockRowSet(descriptor, rows)

    def prolongation(self, assembly: SpaceDescriptor) -> MockOperator:
        return MockOperator.identity(assembly)

    def solve_operator(
        self, op: ActionOperatorProtocol[MockVector, MockFormat]
    ) -> ActionOperatorProtocol[MockVector, MockFormat]:
        return op
