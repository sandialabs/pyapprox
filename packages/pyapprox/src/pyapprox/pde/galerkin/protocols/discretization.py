"""What a steady problem needs of a system on a linear-algebra stack.

A steady residual :math:`R = P F + (I - P)(u - g)` reads the system (its
spatial operator and essential constraints), the stack's state and
parameter spaces, the essential rows as the stack's row set, and the
assembled state Jacobian. It reads no mass, so a discretization of a
steady problem need not have one. Any stack satisfies the protocol: the
SciPy stack of an skfem system
(:class:`~pyapprox.pde.galerkin.time_integration.stack_discretization.GalerkinStackDiscretization`)
and dense backend arrays
(:class:`~pyapprox.pde.galerkin.time_integration.dense_discretization.DenseSteadyDiscretization`).
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pyapprox.pde.galerkin.protocols.system import GalerkinSteadySystemProtocol
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    RowSetProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F_co = TypeVar("F_co", bound=FormatProtocol, covariant=True)


@runtime_checkable
class SteadyDiscretizationProtocol(Protocol, Generic[Array, V, F_co]):
    """A steady system on a stack, without a solver.

    States are backend arrays (``Array``); operators act on the stack's
    vectors (``V``).
    """

    def system(self) -> GalerkinSteadySystemProtocol[Array]:
        """The discretized system."""
        ...

    def nstates(self) -> int:
        """The number of states."""
        ...

    def space(self) -> VectorSpaceProtocol[V]:
        """The state space's vector operations."""
        ...

    def parameter_space(self, nparams: int) -> VectorSpaceProtocol[V]:
        """A space of ``nparams`` parameters on the same stack as the
        states."""
        ...

    def essential_rows(self) -> RowSetProtocol[V]:
        """The essential rows, as the stack's row set of the state space."""
        ...

    def assemble_jacobian(
        self, state: Array, time: float
    ) -> ActionOperatorProtocol[V, F_co]:
        """The state Jacobian at ``(state, time)`` as assembled, essential
        rows included, independent of later calls."""
        ...
