"""The state Jacobian of an skfem system, assembled into a stack.

skfem assembles a whole sparse matrix on one process, so this assembler
takes a
:class:`~pyapprox.pde.assembly.protocols.BulkCapableAssemblyStackProtocol`;
a stack without whole-matrix adds pairs with an element-wise assembler
instead. It satisfies
:class:`~pyapprox.pde.assembly.protocols.JacobianAssemblerProtocol`.
"""

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.pde.assembly.protocols import (
    BulkCapableAssemblyStackProtocol,
    BulkCapableMatrixSinkProtocol,
    BulkMatrixSinkProtocol,
    ElementDofMapProtocol,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    check_state_space,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    SpaceDescriptor,
)

V = TypeVar("V")
F = TypeVar("F")

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


class SkfemJacobianAssembler(Generic[Array, V, F]):
    r"""Assembles :math:`\partial F/\partial u` of an skfem system through
    whole-matrix adds.

    One matrix sink on the state space is fixed at construction, its
    pattern taken from the element DOF maps; each call zeroes it, adds
    skfem's assembled matrix and returns the stored operator. Assembly
    goes through skfem and SciPy, so the operator is not differentiable
    through.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system whose Jacobian is assembled.
    stack : BulkCapableAssemblyStackProtocol
        Where the operator is built; its matrix sinks must accept whole
        assembled matrices.
    dof_map : ElementDofMapProtocol
        The element DOF maps of the system's state (the system's basis).
    descriptor : SpaceDescriptor
        The state space: the operator's domain and range.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol, or the stack's
        sinks lack whole-matrix adds.
    ValueError
        If ``descriptor`` does not have the system's number of states.
    """

    def __init__(
        self,
        system: GalerkinTransientSystemProtocol[Array],
        stack: BulkCapableAssemblyStackProtocol[V, F, _Dofs, _Values],
        dof_map: ElementDofMapProtocol[_Dofs],
        descriptor: SpaceDescriptor,
    ) -> None:
        check_state_space(system, descriptor)
        if not isinstance(stack, BulkCapableAssemblyStackProtocol):
            raise TypeError(
                "stack must satisfy BulkCapableAssemblyStackProtocol, "
                f"got {type(stack).__name__}"
            )
        if not isinstance(dof_map, ElementDofMapProtocol):
            raise TypeError(
                "dof_map must satisfy ElementDofMapProtocol, "
                f"got {type(dof_map).__name__}"
            )
        dofs = dof_map.element_dofs()
        self._sink: BulkCapableMatrixSinkProtocol[V, F, _Dofs, _Values] = (
            stack.matrix_sink(descriptor, descriptor, dofs, dofs)
        )
        # The protocol check cannot see return types, so a stack whose
        # sinks lack whole-matrix adds is refused here, before assembly.
        if not isinstance(self._sink.bulk(), BulkMatrixSinkProtocol):
            raise TypeError(
                f"{type(stack).__name__}'s matrix sinks do not accept whole "
                "assembled matrices; such a stack needs an element-wise "
                "assembler"
            )
        self._spatial_operator = system.spatial_operator()

    def assemble_jacobian(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[V, F]:
        """The spatial Jacobian at ``(state, time)``, independent of
        later calls. Shape: (nstates, nstates)."""
        matrix = require_sparse(self._spatial_operator.spatial_jacobian(state, time))
        self._sink.zero()
        self._sink.bulk().add_assembled(matrix)
        return self._sink.assemble()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sink={self._sink!r})"
