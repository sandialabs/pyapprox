"""The Galerkin state Jacobian as an operator of an injected stack.

``GalerkinJacobianOperator`` assembles the spatial Jacobian of a
Galerkin system, with its essential rows zeroed, into a sink of an
injected :class:`~pyapprox.pde.assembly.protocols.AssemblyStackProtocol`,
so the result is an operator of that stack's own vector and format
types. The sink's pattern is fixed once from the element DOF maps.
"""

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    ElementDofMapProtocol,
    MatrixSinkProtocol,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)

_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]


def require_sparse(matrix: Array) -> spmatrix:
    """Return a Jacobian as the sparse matrix skfem assembles.

    The spatial operators are typed as returning ``Array`` and return a
    SciPy sparse matrix for skfem physics; this is the one place that
    checks it.

    Raises
    ------
    TypeError
        If ``matrix`` is not a SciPy sparse matrix.
    """
    if not isinstance(matrix, spmatrix):
        raise TypeError(
            "the stack Jacobian needs the sparse Jacobian of an skfem "
            f"physics, got {type(matrix).__name__}"
        )
    return matrix


class GalerkinJacobianOperator(Generic[Array, V, F]):
    r"""Assembles :math:`\partial F/\partial u`, essential rows zeroed,
    into an injected stack.

    The essential rows are zeroed by
    :class:`~pyapprox.util.linalg.operators.RowsZeroedOperator` rather
    than in storage, so the stored operator is the plain assembled one.
    The result is labeled with the given linearization point.

    The constraint set gives its DOFs as a backend array; they are
    converted to the stack's index type once, here, as integer data that
    no derivative flows through. Assembly goes through skfem and SciPy,
    so the operator is not differentiable through.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system whose Jacobian is assembled.
    stack : AssemblyStackProtocol
        Where the operator is built.
    dof_map : ElementDofMapProtocol
        The element DOF maps of the system's state, which fix the sink's
        pattern (the system's basis).
    descriptor : SpaceDescriptor
        The state space: the operator's domain and range.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If ``descriptor`` does not have the system's number of states.
    """

    def __init__(
        self,
        system: GalerkinTransientSystemProtocol[Array],
        stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
        dof_map: ElementDofMapProtocol[_Dofs],
        descriptor: SpaceDescriptor,
    ) -> None:
        if not isinstance(system, GalerkinTransientSystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinTransientSystemProtocol, "
                f"got {type(system).__name__}"
            )
        if not isinstance(stack, AssemblyStackProtocol):
            raise TypeError(
                f"stack must satisfy AssemblyStackProtocol, got {type(stack).__name__}"
            )
        if not isinstance(dof_map, ElementDofMapProtocol):
            raise TypeError(
                "dof_map must satisfy ElementDofMapProtocol, "
                f"got {type(dof_map).__name__}"
            )
        if descriptor.size != system.nstates():
            raise ValueError(
                f"descriptor has size {descriptor.size}, the system "
                f"{system.nstates()} states"
            )
        self._spatial_operator = system.spatial_operator()
        dofs = dof_map.element_dofs()
        self._sink: MatrixSinkProtocol[V, F, _Dofs, _Values] = stack.matrix_sink(
            descriptor, descriptor, dofs, dofs
        )
        constraint_set = system.constraint_set()
        essential: _Dofs = np.asarray(
            system.bkd().to_numpy(constraint_set.dofs()), dtype=np.int64
        )
        self._rows: RowSetProtocol[V] = stack.row_set(descriptor, essential)

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        """The Jacobian at ``(state, time)``, labeled with ``point``.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.
        point : LinearizationPoint
            The identity of ``(state, time)``, minted by the caller.

        Returns
        -------
        LinearizedOperator
            The operator, independent of later calls.
        """
        matrix = require_sparse(self._spatial_operator.spatial_jacobian(state, time))
        sink = self._sink
        sink.zero()
        bulk = sink.bulk()
        if bulk is not None:
            bulk.add_assembled(matrix)
        else:
            # Without a bulk capability the nonzeros go in as 1x1 element
            # blocks; they lie in the pattern, as the assembled ones do.
            coo = matrix.tocoo()
            rows: _Dofs = np.asarray(coo.row, dtype=np.int64)[:, None]
            cols: _Dofs = np.asarray(coo.col, dtype=np.int64)[:, None]
            values: _Values = np.asarray(coo.data, dtype=np.float64)[:, None, None]
            sink.add_element_matrices(rows, cols, values)
        return LinearizedOperator(
            RowsZeroedOperator(sink.assemble(), self._rows), point
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sink={self._sink!r})"
