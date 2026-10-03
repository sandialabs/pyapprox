"""Protocols for a composed Galerkin system.

A system is the parts a solver consumes, already composed: the spatial
operator ``F`` (interior plus natural-BC terms), the essential
constraint set, and, for transient problems, the mass. Steady consumers
take ``GalerkinSteadySystemProtocol``, which has no mass; transient
consumers take ``GalerkinTransientSystemProtocol``, which adds it. A
steady problem therefore never carries a placeholder mass whose methods
raise.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.steady_view import SteadyView
from pyapprox.util.backends.protocols import Array, Array_co, Backend


@runtime_checkable
class GalerkinMassProtocol(Protocol, Generic[Array_co]):
    """Supplies the mass matrix ``M`` of ``M du/dt = F(u, t)``.

    A provider rather than a matrix, so a mass that depends on a
    parameter (a density field) is read when needed, never copied stale.
    ``Array`` appears only in return position, so the protocol is
    covariant.
    """

    def mass_matrix(self) -> Array_co:
        """Return the mass matrix. Shape: (nstates, nstates)."""
        ...


@runtime_checkable
class GalerkinSteadySystemProtocol(Protocol, Generic[Array]):
    """The composed parts of a steady Galerkin problem ``F(u) = 0``."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return ``F``, without essential constraints applied."""
        ...

    def constraint_set(self) -> ConstraintSetProtocol[Array]:
        """Return the essential constraints, applied by the consumer."""
        ...

    def steady(self) -> SteadyView[Array]:
        """Return the time-free steady view; raises for time-dependent
        data."""
        ...

    def steady_snapshot(self, time: float) -> SteadyView[Array]:
        """Return the steady view of the data frozen at ``time``."""
        ...


@runtime_checkable
class GalerkinTransientSystemProtocol(
    GalerkinSteadySystemProtocol[Array], Protocol
):
    """The composed parts of a transient problem ``M du/dt = F(u, t)``."""

    def mass_matrix(self) -> Array:
        """Return the mass matrix. Shape: (nstates, nstates)."""
        ...
