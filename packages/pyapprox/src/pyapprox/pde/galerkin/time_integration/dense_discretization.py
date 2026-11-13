"""A steady system whose Jacobian is a dense backend array, on the
backend's dense stack.

For spatial operators that return their Jacobian as a dense backend
array (closed-form or finite-difference operators, built without an
assembly library), :class:`DenseSteadyDiscretization` gives what a
steady residual reads
(:class:`~pyapprox.pde.galerkin.protocols.discretization.SteadyDiscretizationProtocol`):
the backend's dense state and parameter spaces, the essential rows as a
backend row set, and the Jacobian as a dense operator. It works on any
backend, and a Torch graph through the Jacobian's entries survives.
"""

from typing import Generic

from pyapprox.ode.residual_derivatives import backend_spaces, backend_state_space
from pyapprox.pde.galerkin.protocols.system import GalerkinSteadySystemProtocol
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import SpaceDescriptor


class DenseSteadyDiscretization(Generic[Array]):
    """A steady system on the backend's dense stack, without a solver.

    Parameters
    ----------
    system : GalerkinSteadySystemProtocol
        The system; its spatial operator's ``spatial_jacobian`` must
        return a dense ``(nstates, nstates)`` backend array.

    Raises
    ------
    TypeError
        If ``system`` does not satisfy ``GalerkinSteadySystemProtocol``.
    """

    def __init__(self, system: GalerkinSteadySystemProtocol[Array]) -> None:
        if not isinstance(system, GalerkinSteadySystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinSteadySystemProtocol, got "
                f"{type(system).__name__}"
            )
        bkd = system.bkd()
        self._system = system
        self._bkd = bkd
        self._space = backend_state_space(bkd, system.nstates())
        self._descriptor: SpaceDescriptor = self._space.descriptor()
        # Integer index data, which no derivative flows through.
        dofs = [int(dof) for dof in bkd.to_numpy(system.constraint_set().dofs())]
        self._rows: BackendRowSet[Array] = BackendRowSet(self._descriptor, dofs, bkd)

    def system(self) -> GalerkinSteadySystemProtocol[Array]:
        """The discretized system."""
        return self._system

    def nstates(self) -> int:
        """The number of states."""
        return self._descriptor.size

    def space(self) -> BackendVectorSpace[Array]:
        """The state space's vector operations."""
        return self._space

    def parameter_space(self, nparams: int) -> BackendVectorSpace[Array]:
        """A dense space of ``nparams`` parameters in the backend's dtype."""
        _, params = backend_spaces(self._bkd, self.nstates(), nparams)
        return params

    def essential_rows(self) -> BackendRowSet[Array]:
        """The essential rows, as a backend row set of the state space."""
        return self._rows

    def assemble_jacobian(
        self, state: Array, time: float
    ) -> BackendDenseOperator[Array]:
        """The spatial operator's Jacobian at ``(state, time)``, essential
        rows included, as a dense operator."""
        matrix = self._system.spatial_operator().spatial_jacobian(state, time)
        return BackendDenseOperator(matrix, self._descriptor, self._bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self.nstates()})"
