r"""State constraints for Galerkin systems with essential boundary data.

The time integrator applies a
:class:`~pyapprox.ode.protocols.StateConstraintProtocol` to every stored
state. For a Galerkin system with essential (Dirichlet) data :math:`g(t)`
on the DOFs :math:`D`:

- :class:`EssentialPin` sets :math:`y_D = g(t)` and leaves the other
  entries unchanged. With the symmetric row treatment the stored states
  are exact on the boundary, and total content is conserved to the
  stepper's order.

The constraints are affine in the state and independent of the
parameters, so their parameter and second derivatives are stated zeros
and the sensitivity sweeps apply only their tangents.
"""

from typing import Generic

from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_spaces
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.linalg.operators import (
    IdentityOperator,
    MatrixFree,
    RowsZeroedOperator,
)


class _PinTangent(Generic[Array]):
    """The pin's tangent, the identity with the essential rows zeroed,
    the same at every state and time."""

    def __init__(self, op: RowsZeroedOperator[Array, MatrixFree]) -> None:
        self._op = op

    def __call__(
        self, state: Array, time: float
    ) -> RowsZeroedOperator[Array, MatrixFree]:
        return self._op


class EssentialPin(Generic[Array]):
    r""":math:`\Pi_t(y)`: ``y`` with its essential entries set to the
    boundary data :math:`g(t)`, the other entries unchanged.

    A projection, affine in the state: its tangent is the identity with
    the essential rows zeroed (so :math:`\Pi'^\top` zeros the essential
    entries of an adjoint), and its parameter and second derivatives are
    stated zeros.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The system whose constraint set gives the essential DOFs and data.
    nparams : int
        The residual's number of parameters, sizing the derivatives'
        parameter space. The boundary data do not depend on them.
    """

    def __init__(
        self, system: GalerkinTransientSystemProtocol[Array], nparams: int
    ) -> None:
        self._constraint_set = system.constraint_set()
        bkd = system.bkd()
        states, params = backend_spaces(bkd, system.nstates(), nparams)
        dofs = [int(dof) for dof in bkd.to_numpy(self._constraint_set.dofs())]
        rows = BackendRowSet(states.descriptor(), dofs, bkd)
        tangent = _PinTangent(RowsZeroedOperator(IdentityOperator(states), rows))
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.second_order(
                states,
                params,
                tangent,
                param=STATED_ZERO,
                state_state=STATED_ZERO,
                state_param=STATED_ZERO,
                param_param=STATED_ZERO,
            )
        )

    def constrain(self, state: Array, time: float) -> Array:
        """``state`` with :math:`y_D = g(t)`."""
        return self._constraint_set.inject(state, time)

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ndofs={self._constraint_set.ndofs()})"
