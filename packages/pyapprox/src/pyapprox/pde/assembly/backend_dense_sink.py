r"""Matrix assembly into dense backend arrays.

The matrix sink accumulates element blocks into a dense backend array
and assembles a
:class:`~pyapprox.util.linalg.backend_dense.BackendDenseOperator`. Each
add is a backend expression (the blocks are placed by products with
constant 0/1 arrays, never written in place), so a torch graph through
the element values survives assembly. Its pattern is every entry, so it
suits small operators: interface ports, verification oracles.
"""

from typing import Generic, Optional

from pyapprox.pde.assembly.protocols import BulkMatrixSinkProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    require_backend_dtype,
)
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor


def _check_dofs(dofs: Array, size: int, name: str, bkd: Backend[Array]) -> int:
    """``dofs``'s number of elements, after checking its shape and range."""
    if dofs.ndim != 2:
        raise ValueError(f"{name} must have shape (nsel, n), got {tuple(dofs.shape)}")
    if dofs.shape[0] and dofs.shape[1]:
        if bkd.to_int(bkd.min(dofs)) < 0 or bkd.to_int(bkd.max(dofs)) >= size:
            raise ValueError(f"{name} must lie in [0, {size})")
    return int(dofs.shape[0])


class BackendDenseMatrixSink(Generic[Array]):
    """A matrix sink accumulating into a dense backend array.

    Satisfies :class:`~pyapprox.pde.assembly.protocols.MatrixSinkProtocol`
    with backend arrays as DOF maps and element values. The pattern is
    every entry of the matrix, so no in-range add is refused; the DOF maps
    given at construction are only checked. Each add costs
    O(nsel * nrow * ncol * (range.size + domain.size)) through the
    placement products.

    Parameters
    ----------
    range, domain
        The spaces of the rows and the columns; both must name the
        backend's configured dtype.
    row_dofs, col_dofs
        Element DOF maps, int64, shapes (nsel, nrow) and (nsel, ncol).
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If a descriptor's dtype is not the backend's, a DOF map has the
        wrong shape or lies outside its space, or the maps list different
        numbers of elements.
    """

    def __init__(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: Array,
        col_dofs: Array,
        bkd: Backend[Array],
    ) -> None:
        require_backend_dtype(range, bkd, "BackendDenseMatrixSink")
        require_backend_dtype(domain, bkd, "BackendDenseMatrixSink")
        nrows = _check_dofs(row_dofs, range.size, "row_dofs", bkd)
        ncols = _check_dofs(col_dofs, domain.size, "col_dofs", bkd)
        if nrows != ncols:
            raise ValueError(
                "row_dofs and col_dofs must list the same elements, got "
                f"{nrows} and {ncols}"
            )
        self._range, self._domain, self._bkd = range, domain, bkd
        dtype = bkd.default_dtype()
        self._row_placement = bkd.eye(range.size, dtype=dtype)
        self._col_placement = bkd.eye(domain.size, dtype=dtype)
        self._matrix = bkd.zeros((range.size, domain.size), dtype=dtype)

    def add_element_matrices(
        self,
        row_dofs: Array,
        col_dofs: Array,
        values: Array,
        weights: Optional[Array] = None,
    ) -> None:
        bkd = self._bkd
        nsel = _check_dofs(row_dofs, self._range.size, "row_dofs", bkd)
        if _check_dofs(col_dofs, self._domain.size, "col_dofs", bkd) != nsel:
            raise ValueError("row_dofs and col_dofs must list the same elements")
        expected = (nsel, int(row_dofs.shape[1]), int(col_dofs.shape[1]))
        if tuple(values.shape) != expected:
            raise ValueError(
                f"values must have shape {expected}, got {tuple(values.shape)}"
            )
        if weights is not None:
            if tuple(weights.shape) != (nsel,):
                raise ValueError(
                    f"weights must have shape ({nsel},), got {tuple(weights.shape)}"
                )
            values = values * weights[:, None, None]
        rows = self._row_placement[row_dofs]
        cols = self._col_placement[col_dofs]
        self._matrix = self._matrix + bkd.einsum("eri,erc,ecj->ij", rows, values, cols)

    def bulk(self) -> Optional[BulkMatrixSinkProtocol]:
        """None: whole sparse matrices are not accepted."""
        return None

    def zero(self) -> None:
        self._matrix = self._bkd.zeros(
            (self._range.size, self._domain.size), dtype=self._bkd.default_dtype()
        )

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(
            self._matrix,
            self._domain,
            self._bkd,
            range=self._range,
            properties=properties,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape=({self._range.size}, "
            f"{self._domain.size}))"
        )


class BackendDenseMatrixAssembly(Generic[Array]):
    """Matrix sinks of dense backend arrays.

    Provides only ``matrix_sink``: it satisfies a consumer that asks for
    matrix assembly alone, such as
    :class:`~pyapprox.pde.coupling.ports.PortAssemblyProtocol`, and is not
    a whole :class:`~pyapprox.pde.assembly.protocols.AssemblyStackProtocol`.

    Parameters
    ----------
    bkd
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: Array,
        col_dofs: Array,
    ) -> BackendDenseMatrixSink[Array]:
        return BackendDenseMatrixSink(range, domain, row_dofs, col_dofs, self._bkd)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
