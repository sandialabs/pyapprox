"""A transient Galerkin system assembled into a linear-algebra stack.

:class:`GalerkinStackDiscretization` holds what time integration needs
of a system on a stack: the stack's state space, the essential rows as
the stack's row set, the state Jacobian with those rows zeroed, and the
mass with those rows replaced by identity rows. It is generic over the
stack and holds no solver: which mass (consistent or lumped) and how
entries reach the stack are the injected assemblers' choices, and how
the resulting systems are solved is the caller's. Builders bind it to a
particular assembler and stack, for instance
:func:`~pyapprox.pde.galerkin.time_integration.skfem_discretization.skfem_discretization`.

Explicit forward stepping needs only the mass; the Jacobian is needed by
implicit steps and by every derivative (sensitivities, adjoints and
Hessian-vector products, explicit methods included). It is required
here, and assembled only when asked for.
"""

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    JacobianAssemblerProtocol,
    MassAssemblerProtocol,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
    check_state_space,
    essential_rows,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    RowReplacedOperator,
    RowSetProtocol,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)

_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]


class GalerkinStackDiscretization(Generic[Array, V, F]):
    """A transient Galerkin system on a stack, without a solver.

    States are backend arrays (``Array``) and operators act on the
    stack's vectors (``V``). A system that steps on these operators
    needs the two to coincide, as they do on the SciPy stack, whose
    vectors are NumPy arrays; a builder pairs them.

    The essential rows come from the system's constraint set, once, as
    the stack's row set, so every
    consumer (the forward residual, a boundary-condition wrapper, the
    sensitivity and adjoint sweeps) edits the same rows, and an adjoint
    solve is the exact transpose of the forward one.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system.
    stack : AssemblyStackProtocol
        The stack the assemblers build in.
    jacobian_assembler : JacobianAssemblerProtocol
        Assembles the state Jacobian into ``stack``.
    mass_assembler : MassAssemblerProtocol
        Assembles the mass into ``stack``, consistent or lumped.
    descriptor : SpaceDescriptor
        The state space.

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
        jacobian_assembler: JacobianAssemblerProtocol[Array, V, F],
        mass_assembler: MassAssemblerProtocol[V, F],
        descriptor: SpaceDescriptor,
    ) -> None:
        check_state_space(system, descriptor)
        if not isinstance(mass_assembler, MassAssemblerProtocol):
            raise TypeError(
                "mass_assembler must satisfy MassAssemblerProtocol, "
                f"got {type(mass_assembler).__name__}"
            )
        self._system = system
        self._descriptor = descriptor
        self._rows: RowSetProtocol[V] = essential_rows(system, stack, descriptor)
        self._space: VectorSpaceProtocol[V] = stack.space(descriptor)
        self._jacobian = GalerkinJacobianOperator(jacobian_assembler, self._rows)
        self._mass_assembler = mass_assembler

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """The discretized system."""
        return self._system

    def nstates(self) -> int:
        """The number of states."""
        return self._descriptor.size

    def descriptor(self) -> SpaceDescriptor:
        """The state space's descriptor."""
        return self._descriptor

    def space(self) -> VectorSpaceProtocol[V]:
        """The state space's vector operations."""
        return self._space

    def essential_rows(self) -> RowSetProtocol[V]:
        """The essential rows, as the stack's row set of the state space."""
        return self._rows

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        """The state Jacobian at ``(state, time)`` with the essential rows
        zeroed, labeled with ``point``, independent of later calls."""
        return self._jacobian.jacobian_operator(state, time, point)

    def unconstrained_mass(self) -> ActionOperatorProtocol[V, F]:
        """The mass as assembled, essential rows included: for quantities
        read on those rows, such as a boundary reaction."""
        return self._mass_assembler.assemble_mass()

    def mass(self) -> RowReplacedOperator[V, F]:
        """The mass with the essential rows replaced by identity rows: the
        mass time integration steps with."""
        return RowReplacedOperator(self.unconstrained_mass(), self._rows)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nstates={self.nstates()}, "
            f"mass_assembler={self._mass_assembler!r})"
        )
