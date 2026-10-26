"""The mass of an skfem system, assembled into a stack.

The consistent mass goes in as skfem's whole matrix, so this assembler
takes a
:class:`~pyapprox.pde.assembly.protocols.BulkCapableAssemblyStackProtocol`,
as :class:`SkfemJacobianAssembler` does. The lumped mass, the row sums
of the consistent one, goes in as one-entry element blocks on the
diagonal, so it needs no capability of the stack beyond element adds. It
satisfies :class:`~pyapprox.pde.assembly.protocols.MassAssemblerProtocol`.
"""

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly.protocols import (
    BulkCapableAssemblyStackProtocol,
    BulkCapableMatrixSinkProtocol,
    BulkMatrixSinkProtocol,
    ElementDofMapProtocol,
    MatrixSinkProtocol,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    check_state_space,
)
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    require_sparse,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    OperatorProperties,
    SpaceDescriptor,
)

V = TypeVar("V")
F = TypeVar("F")

_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]


class SkfemMassAssembler(Generic[Array, V, F]):
    """Assembles the mass of an skfem system, consistent or lumped.

    The consistent mass is declared symmetric, the lumped one symmetric
    and diagonal; neither is declared SPD, since a mixed system's mass
    (Stokes) is singular and a lumped quadratic one can have nonpositive
    row sums. Assembly goes through skfem and SciPy, so the operator is
    not differentiable through. The essential rows are left as assembled:
    a time integrator replaces them, for instance with
    :class:`~pyapprox.util.linalg.operators.RowReplacedOperator`.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system whose mass is assembled.
    stack : BulkCapableAssemblyStackProtocol
        Where the operator is built; its matrix sinks must accept whole
        assembled matrices.
    dof_map : ElementDofMapProtocol
        The element DOF maps of the system's state (the system's basis),
        the consistent mass's pattern.
    descriptor : SpaceDescriptor
        The state space: the operator's domain and range.
    lumped : bool, default False
        If True, assemble the row-sum lumped mass instead.

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
        lumped: bool = False,
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
        self._system = system
        self._lumped = lumped
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
        self._diagonal: _Dofs = np.arange(descriptor.size, dtype=np.int64)[:, None]
        self._lumped_sink: MatrixSinkProtocol[V, F, _Dofs, _Values] = stack.matrix_sink(
            descriptor, descriptor, self._diagonal, self._diagonal
        )

    def assemble_mass(self) -> ActionOperatorProtocol[V, F]:
        """The mass, consistent or lumped as constructed, independent of
        later calls. Shape: (nstates, nstates)."""
        matrix = require_sparse(self._system.mass_matrix(), "mass")
        if not self._lumped:
            self._sink.zero()
            self._sink.bulk().add_assembled(matrix)
            return self._sink.assemble(OperatorProperties(symmetric=True))
        row_sums: _Values = np.asarray(matrix.sum(axis=1), dtype=np.float64)
        self._lumped_sink.zero()
        self._lumped_sink.add_element_matrices(
            self._diagonal, self._diagonal, row_sums[:, :, None]
        )
        return self._lumped_sink.assemble(
            OperatorProperties(symmetric=True, diagonal=True)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lumped={self._lumped})"
