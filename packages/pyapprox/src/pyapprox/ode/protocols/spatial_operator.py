"""Protocol for a spatial operator: the right-hand side of M du/dt = F(u, t).

A spatial operator is ``F(u, t)`` and its state Jacobian, with no
row replacement for essential constraints. It is solver-neutral: a
discretized PDE supplies one (for Galerkin, the composition of the
interior operator with the natural-BC terms), and so can any other
semi-discrete system. Consumers that need only ``F`` (a second-order
reduction using it as a restoring force, say) depend on this protocol
alone.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SpatialOperatorProtocol(Protocol, Generic[Array]):
    """``F(u, t)`` and ``dF/du`` of ``M du/dt = F(u, t)``."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute ``F(u, t)``. Shape: (nstates,)."""
        ...

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF/du``. Shape: (nstates, nstates); may be sparse."""
        ...

    def state_derivatives(self) -> StateDerivatives[Array]:
        """Return the optional second state derivatives of ``F``.

        Consumers choose their derivative tier from this bundle, once.
        """
        ...

    def is_time_invariant(self) -> bool:
        """Whether ``F(u, t)`` is DECLARED independent of ``t``.

        Aggregated from the declarations of the data ``F`` holds
        (coefficients, forcing, boundary-term data). A steady consumer
        may evaluate ``F`` at any time only when this holds.
        """
        ...
