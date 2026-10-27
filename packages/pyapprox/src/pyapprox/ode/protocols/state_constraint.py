r"""Protocol for a constraint the time integrator applies to stored states.

The integrator applies :math:`\Pi_t` to the initial state and to each step
result, and its sensitivity sweeps apply the derivatives the constraint
declares. How the constraint is computed is the discretization's choice.
"""

from typing import Protocol, runtime_checkable

from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class StateConstraintProtocol(Protocol[Array]):
    """A map from a state to the state satisfying the constraints.

    States are 1-D arrays of length ``nstates``.
    """

    def constrain(self, state: Array, time: float) -> Array:
        r""":math:`\Pi_t(y)`: the state satisfying the constraints at
        ``time``."""
        ...

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        """The constraint's optional derivatives."""
        ...
