r"""The single-process NumPy/SciPy assembly stack.

Vectors are NumPy blocks of shape ``(n, ncols)``, operators are
:class:`~pyapprox.util.linalg.scipy_stack.ScipyCSROperator`, element DOF
maps are int64 arrays and element values arrays of the stack's scalar
type ``S``, its configured precision. A sink's pattern is a sorted array
of ``row * ncols + col`` keys built once from the element DOF maps; each
add locates its entries with one vectorized search and scatter-adds them,
with no per-element Python loop.
"""

from typing import Generic, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse, spmatrix

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import (
    CSRFormat,
    NumpyRowSet,
    NumpyVectorSpace,
    S,
    ScipyCSROperator,
    require_scalar,
)

_Dofs = NDArray[np.int64]


def _check_dofs(dofs: _Dofs, size: int, name: str) -> NDArray[np.intp]:
    if dofs.ndim != 2:
        raise ValueError(f"{name} must have shape (nsel, n), got {dofs.shape}")
    as_int = np.asarray(dofs, dtype=np.intp)
    if as_int.size and (as_int.min() < 0 or as_int.max() >= size):
        raise ValueError(f"{name} must lie in [0, {size})")
    return as_int


def _element_keys(
    row_dofs: NDArray[np.intp], col_dofs: NDArray[np.intp], ncols: int
) -> NDArray[np.intp]:
    """The ``row * ncols + col`` key of every entry, shape (nsel, nrow, ncol)."""
    return row_dofs[:, :, None] * ncols + col_dofs[:, None, :]


def _element_weights(weights: Optional[NDArray[S]], nsel: int) -> Optional[NDArray[S]]:
    if weights is None:
        return None
    if weights.shape != (nsel,):
        raise ValueError(f"weights must have shape ({nsel},), got {weights.shape}")
    return weights


class _ScipyBulkMatrixSink(Generic[S]):
    def __init__(self, sink: "ScipyMatrixSink[S]") -> None:
        self._sink: ScipyMatrixSink[S] = sink

    def add_assembled(self, matrix: spmatrix) -> None:
        self._sink._add_assembled(matrix)


class ScipyMatrixSink(Generic[S]):
    """A matrix sink with a fixed CSR pattern.

    Satisfies :class:`~pyapprox.pde.assembly.protocols.MatrixSinkProtocol`.
    The pattern is the union of every element block given at
    construction; adds outside it raise rather than reallocate.

    Parameters
    ----------
    range, domain : SpaceDescriptor
        The assembly spaces of the rows and the columns; both must name
        ``scalar``.
    row_dofs, col_dofs : ndarray of int64
        Element DOF maps, shapes (nsel, nrow) and (nsel, ncol).
    scalar : type
        The configured scalar type.
    """

    def __init__(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
        scalar: type[S],
    ) -> None:
        require_scalar(range, scalar, "ScipyMatrixSink")
        require_scalar(domain, scalar, "ScipyMatrixSink")
        rows = _check_dofs(row_dofs, range.size, "row_dofs")
        cols = _check_dofs(col_dofs, domain.size, "col_dofs")
        if rows.shape[0] != cols.shape[0]:
            raise ValueError(
                f"row_dofs and col_dofs must list the same elements, got "
                f"{rows.shape[0]} and {cols.shape[0]}"
            )
        self._range = range
        self._domain = domain
        self._scalar: type[S] = scalar
        self._keys = np.unique(_element_keys(rows, cols, domain.size))
        pattern_rows = self._keys // domain.size
        self._indices = (self._keys % domain.size).astype(np.intp)
        self._indptr = np.searchsorted(
            pattern_rows, np.arange(range.size + 1), side="left"
        ).astype(np.intp)
        self._data: NDArray[S] = np.zeros(self._keys.shape[0], dtype=scalar)

    def nnz(self) -> int:
        """The number of entries in the pattern."""
        return int(self._keys.shape[0])

    def _positions(self, keys: NDArray[np.intp]) -> NDArray[np.intp]:
        positions = np.searchsorted(self._keys, keys)
        inside = positions < self._keys.shape[0]
        inside[inside] = self._keys[positions[inside]] == keys[inside]
        if not np.all(inside):
            raise ValueError(
                f"{int(np.count_nonzero(~inside))} entries lie outside the sink's "
                "sparsity pattern"
            )
        return positions

    def add_element_matrices(
        self,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
        values: NDArray[S],
        weights: Optional[NDArray[S]] = None,
    ) -> None:
        rows = _check_dofs(row_dofs, self._range.size, "row_dofs")
        cols = _check_dofs(col_dofs, self._domain.size, "col_dofs")
        expected = (rows.shape[0], rows.shape[1], cols.shape[1])
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}, got {values.shape}")
        scale = _element_weights(weights, rows.shape[0])
        contributions = values if scale is None else values * scale[:, None, None]
        positions = self._positions(_element_keys(rows, cols, self._domain.size))
        np.add.at(self._data, positions.ravel(), contributions.ravel())

    def _add_assembled(self, matrix: spmatrix) -> None:
        if not issparse(matrix):
            raise TypeError(
                "add_assembled needs a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        shape = (self._range.size, self._domain.size)
        if matrix.shape != shape:
            raise ValueError(f"matrix must have shape {shape}, got {matrix.shape}")
        coo = matrix.tocoo()
        keys = coo.row.astype(np.intp) * self._domain.size + coo.col.astype(np.intp)
        np.add.at(self._data, self._positions(keys), coo.data)

    def bulk(self) -> _ScipyBulkMatrixSink[S]:
        """Always available: this stack is single-process."""
        return _ScipyBulkMatrixSink(self)

    def zero(self) -> None:
        self._data[:] = 0.0

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> ScipyCSROperator[S]:
        matrix = csr_matrix(
            (self._data.copy(), self._indices, self._indptr),
            shape=(self._range.size, self._domain.size),
        )
        return ScipyCSROperator(
            matrix, self._domain, self._range, properties, scalar=self._scalar
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nnz={self.nnz()})"


class _ScipyBulkVectorSink(Generic[S]):
    def __init__(self, sink: "ScipyVectorSink[S]") -> None:
        self._sink: ScipyVectorSink[S] = sink

    def add_assembled(self, vector: NDArray[S]) -> None:
        self._sink._add_assembled(vector)


class ScipyVectorSink(Generic[S]):
    """A vector sink accumulating into a NumPy column.

    Satisfies :class:`~pyapprox.pde.assembly.protocols.VectorSinkProtocol`.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The assembly space of the vector; it must name ``scalar``.
    scalar : type
        The configured scalar type.
    """

    def __init__(self, descriptor: SpaceDescriptor, scalar: type[S]) -> None:
        require_scalar(descriptor, scalar, "ScipyVectorSink")
        self._descriptor = descriptor
        self._data: NDArray[S] = np.zeros(descriptor.size, dtype=scalar)

    def add_element_vectors(
        self,
        row_dofs: _Dofs,
        values: NDArray[S],
        weights: Optional[NDArray[S]] = None,
    ) -> None:
        rows = _check_dofs(row_dofs, self._descriptor.size, "row_dofs")
        if values.shape != rows.shape:
            raise ValueError(f"values must have shape {rows.shape}, got {values.shape}")
        scale = _element_weights(weights, rows.shape[0])
        contributions = values if scale is None else values * scale[:, None]
        np.add.at(self._data, rows.ravel(), contributions.ravel())

    def _add_assembled(self, vector: NDArray[S]) -> None:
        if vector.shape != self._data.shape:
            raise ValueError(
                f"vector must have shape {self._data.shape}, got {vector.shape}"
            )
        self._data += vector

    def bulk(self) -> _ScipyBulkVectorSink[S]:
        """Always available: this stack is single-process."""
        return _ScipyBulkVectorSink(self)

    def zero(self) -> None:
        self._data[:] = 0.0

    def assemble(self) -> NDArray[S]:
        return self._data.copy()[:, None]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._descriptor!r})"


class ScipyAssemblyStack(Generic[S]):
    """Single-process assembly into NumPy blocks and CSR operators.

    Satisfies
    :class:`~pyapprox.pde.assembly.protocols.AssemblyStackProtocol` with
    ``NDArray[S]`` as vectors and element values, int64 arrays as DOF
    maps, and :class:`~pyapprox.util.linalg.scipy_stack.CSRFormat`
    operators.

    Parameters
    ----------
    scalar : type
        The configured scalar type (precision) of every vector and
        operator the stack builds.
    prolongations : mapping, optional
        For an assembly space whose prolongation is not the identity, the
        solve space and :math:`P` (a sparse matrix of shape
        ``(assembly.size, solve.size)`` and dtype ``scalar``), keyed by the
        assembly space. Every other space is its own solve space, with
        :math:`P = I`.

    Raises
    ------
    ValueError
        If a prolongation's shape disagrees with its spaces.
    """

    def __init__(
        self,
        scalar: type[S],
        prolongations: Optional[
            Mapping[SpaceDescriptor, tuple[SpaceDescriptor, spmatrix]]
        ] = None,
    ) -> None:
        self._scalar: type[S] = scalar
        self._prolongations: dict[SpaceDescriptor, ScipyCSROperator[S]] = {}
        for assembly, (solve, matrix) in (prolongations or {}).items():
            if matrix.shape != (assembly.size, solve.size):
                raise ValueError(
                    f"prolongation of {assembly.name!r} must have shape "
                    f"({assembly.size}, {solve.size}), got {matrix.shape}"
                )
            self._prolongations[assembly] = ScipyCSROperator(
                matrix, solve, assembly, scalar=scalar
            )

    def space(self, descriptor: SpaceDescriptor) -> NumpyVectorSpace[S]:
        return NumpyVectorSpace(descriptor, self._scalar)

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Dofs,
        col_dofs: _Dofs,
    ) -> ScipyMatrixSink[S]:
        return ScipyMatrixSink(range, domain, row_dofs, col_dofs, self._scalar)

    def vector_sink(self, descriptor: SpaceDescriptor) -> ScipyVectorSink[S]:
        return ScipyVectorSink(descriptor, self._scalar)

    def row_set(self, descriptor: SpaceDescriptor, rows: _Dofs) -> NumpyRowSet:
        return NumpyRowSet(descriptor, rows)

    def prolongation(self, assembly: SpaceDescriptor) -> ScipyCSROperator[S]:
        if assembly in self._prolongations:
            return self._prolongations[assembly]
        return ScipyCSROperator.identity(assembly, self._scalar)

    def solve_operator(
        self, op: ActionOperatorProtocol[NDArray[S], CSRFormat]
    ) -> ActionOperatorProtocol[NDArray[S], CSRFormat]:
        r""":math:`P_r^\top A P_d`, or ``op`` itself when both are the
        identity.

        Symmetry is kept (:math:`P^\top A P` is symmetric when :math:`A`
        is and both sides share :math:`P`); SPD is not declared, since it
        also needs :math:`P` of full column rank.
        """
        if (
            op.range() not in self._prolongations
            and op.domain() not in self._prolongations
        ):
            return op
        p_range = self.prolongation(op.range())
        p_domain = self.prolongation(op.domain())
        matrix = (
            p_range.format().matrix().T
            @ op.format().matrix()
            @ p_domain.format().matrix()
        ).tocsr()
        symmetric = op.properties().symmetric and op.range() == op.domain()
        return ScipyCSROperator(
            matrix,
            p_domain.domain(),
            p_range.domain(),
            OperatorProperties(symmetric=symmetric),
            scalar=self._scalar,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nprolongations={len(self._prolongations)})"
