r"""The single-process NumPy/SciPy assembly stack.

Vectors are NumPy blocks of shape ``(n, ncols)``, operators are
:class:`~pyapprox.util.linalg.scipy_stack.ScipyCSROperator`, and element
DOF maps and values are NumPy arrays. A sink's pattern is a sorted array
of ``row * ncols + col`` keys built once from the element DOF maps; each
add locates its entries with one vectorized search and scatter-adds them,
with no per-element Python loop.
"""

from typing import Any, Mapping, Optional

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
    NumpyVectorSpace,
    ScipyCSROperator,
)

_Array = NDArray[Any]


def _check_dofs(dofs: _Array, size: int, name: str) -> NDArray[np.intp]:
    if dofs.ndim != 2:
        raise ValueError(f"{name} must have shape (nsel, n), got {dofs.shape}")
    as_int = np.asarray(dofs, dtype=np.intp)
    if as_int.size and (as_int.min() < 0 or as_int.max() >= size):
        raise ValueError(f"{name} must lie in [0, {size})")
    return as_int


def _element_keys(
    row_dofs: NDArray[np.intp], col_dofs: NDArray[np.intp], ncols: int
) -> _Array:
    """The ``row * ncols + col`` key of every entry, shape (nsel, nrow, ncol)."""
    return row_dofs[:, :, None] * ncols + col_dofs[:, None, :]


def _element_weights(weights: Optional[_Array], nsel: int) -> Optional[_Array]:
    if weights is None:
        return None
    if weights.shape != (nsel,):
        raise ValueError(f"weights must have shape ({nsel},), got {weights.shape}")
    return weights


class _ScipyBulkMatrixSink:
    def __init__(self, sink: "ScipyMatrixSink") -> None:
        self._sink = sink

    def add_assembled(self, matrix: spmatrix) -> None:
        self._sink._add_assembled(matrix)


class ScipyMatrixSink:
    """A matrix sink with a fixed CSR pattern.

    Satisfies :class:`~pyapprox.pde.assembly.protocols.MatrixSinkProtocol`.
    The pattern is the union of every element block given at
    construction; adds outside it raise rather than reallocate.

    Parameters
    ----------
    range, domain : SpaceDescriptor
        The assembly spaces of the rows and the columns.
    row_dofs, col_dofs : ndarray of int
        Element DOF maps, shapes (nsel, nrow) and (nsel, ncol).
    """

    def __init__(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Array,
        col_dofs: _Array,
    ) -> None:
        rows = _check_dofs(row_dofs, range.size, "row_dofs")
        cols = _check_dofs(col_dofs, domain.size, "col_dofs")
        if rows.shape[0] != cols.shape[0]:
            raise ValueError(
                f"row_dofs and col_dofs must list the same elements, got "
                f"{rows.shape[0]} and {cols.shape[0]}"
            )
        self._range = range
        self._domain = domain
        self._keys = np.unique(_element_keys(rows, cols, domain.size))
        pattern_rows = self._keys // domain.size
        self._indices = (self._keys % domain.size).astype(np.intp)
        self._indptr = np.searchsorted(
            pattern_rows, np.arange(range.size + 1), side="left"
        ).astype(np.intp)
        self._data = np.zeros(self._keys.shape[0], dtype=np.dtype(range.dtype))

    def nnz(self) -> int:
        """The number of entries in the pattern."""
        return int(self._keys.shape[0])

    def _positions(self, keys: _Array) -> NDArray[np.intp]:
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
        row_dofs: _Array,
        col_dofs: _Array,
        values: _Array,
        weights: Optional[_Array] = None,
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

    def bulk(self) -> _ScipyBulkMatrixSink:
        """Always available: this stack is single-process."""
        return _ScipyBulkMatrixSink(self)

    def zero(self) -> None:
        self._data[:] = 0.0

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> ScipyCSROperator:
        matrix = csr_matrix(
            (self._data.copy(), self._indices, self._indptr),
            shape=(self._range.size, self._domain.size),
        )
        return ScipyCSROperator(matrix, self._domain, self._range, properties)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nnz={self.nnz()})"


class _ScipyBulkVectorSink:
    def __init__(self, sink: "ScipyVectorSink") -> None:
        self._sink = sink

    def add_assembled(self, vector: _Array) -> None:
        self._sink._add_assembled(vector)


class ScipyVectorSink:
    """A vector sink accumulating into a NumPy column.

    Satisfies :class:`~pyapprox.pde.assembly.protocols.VectorSinkProtocol`.

    Parameters
    ----------
    descriptor : SpaceDescriptor
        The assembly space of the vector.
    """

    def __init__(self, descriptor: SpaceDescriptor) -> None:
        self._descriptor = descriptor
        self._data = np.zeros(descriptor.size, dtype=np.dtype(descriptor.dtype))

    def add_element_vectors(
        self, row_dofs: _Array, values: _Array, weights: Optional[_Array] = None
    ) -> None:
        rows = _check_dofs(row_dofs, self._descriptor.size, "row_dofs")
        if values.shape != rows.shape:
            raise ValueError(f"values must have shape {rows.shape}, got {values.shape}")
        scale = _element_weights(weights, rows.shape[0])
        contributions = values if scale is None else values * scale[:, None]
        np.add.at(self._data, rows.ravel(), contributions.ravel())

    def _add_assembled(self, vector: _Array) -> None:
        if vector.shape != self._data.shape:
            raise ValueError(
                f"vector must have shape {self._data.shape}, got {vector.shape}"
            )
        self._data += vector

    def bulk(self) -> _ScipyBulkVectorSink:
        """Always available: this stack is single-process."""
        return _ScipyBulkVectorSink(self)

    def zero(self) -> None:
        self._data[:] = 0.0

    def assemble(self) -> _Array:
        return self._data.copy()[:, None]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._descriptor!r})"


class ScipyAssemblyStack:
    """Single-process assembly into NumPy blocks and CSR operators.

    Satisfies
    :class:`~pyapprox.pde.assembly.protocols.AssemblyStackProtocol` with
    NumPy arrays as vectors and element data, and
    :class:`~pyapprox.util.linalg.scipy_stack.CSRFormat` operators.

    Parameters
    ----------
    prolongations : mapping, optional
        For an assembly space whose prolongation is not the identity, the
        solve space and :math:`P` (a sparse matrix of shape
        ``(assembly.size, solve.size)``), keyed by the assembly space.
        Every other space is its own solve space, with :math:`P = I`.

    Raises
    ------
    ValueError
        If a prolongation's shape disagrees with its spaces.
    """

    def __init__(
        self,
        prolongations: Optional[
            Mapping[SpaceDescriptor, tuple[SpaceDescriptor, spmatrix]]
        ] = None,
    ) -> None:
        self._prolongations: dict[SpaceDescriptor, ScipyCSROperator] = {}
        for assembly, (solve, matrix) in (prolongations or {}).items():
            if matrix.shape != (assembly.size, solve.size):
                raise ValueError(
                    f"prolongation of {assembly.name!r} must have shape "
                    f"({assembly.size}, {solve.size}), got {matrix.shape}"
                )
            self._prolongations[assembly] = ScipyCSROperator(matrix, solve, assembly)

    def space(self, descriptor: SpaceDescriptor) -> NumpyVectorSpace:
        return NumpyVectorSpace(descriptor)

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: _Array,
        col_dofs: _Array,
    ) -> ScipyMatrixSink:
        return ScipyMatrixSink(range, domain, row_dofs, col_dofs)

    def vector_sink(self, descriptor: SpaceDescriptor) -> ScipyVectorSink:
        return ScipyVectorSink(descriptor)

    def prolongation(self, assembly: SpaceDescriptor) -> ScipyCSROperator:
        if assembly in self._prolongations:
            return self._prolongations[assembly]
        return ScipyCSROperator.identity(assembly)

    def solve_operator(
        self, op: ActionOperatorProtocol[_Array, CSRFormat]
    ) -> ActionOperatorProtocol[_Array, CSRFormat]:
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
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nprolongations={len(self._prolongations)})"
