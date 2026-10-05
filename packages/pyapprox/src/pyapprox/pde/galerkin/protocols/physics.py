"""Physics protocols for Galerkin finite element methods.

A Galerkin physics is its interior weak form ``F_Omega`` and its mass
``M``; boundary conditions are not part of it. ``compose_galerkin_system``
joins a physics with its boundary conditions into the system that models
and solvers consume, ``M du/dt = F(u, t)`` with ``F = F_Omega + F_Gamma``.

Parameter sensitivity (param_jacobian, HVP) is handled by the separate
ParameterizationProtocol layer, not embedded in physics.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class GalerkinInteriorOperatorProtocol(Protocol, Generic[Array]):
    """The interior part of the spatial operator, ``F_Omega``.

    A physics' own mathematics: the weak form over the domain, with no
    boundary condition in it. The spatial operator is composed as
    ``F = F_Omega + F_Gamma``, where the natural-BC part ``F_Gamma`` is
    added once, by ``pyapprox.pde.boundary.NaturalBCOperator``, and never
    by the physics itself.

    Anything implementing this protocol (a physics, or an operator such
    as a prior's precision) gets the composed ``F`` from
    ``ComposedSpatialOperator``, with no base class required.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute ``F_Omega(u, t)``. Shape: (nstates,)."""
        ...

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF_Omega/du``. Shape: (nstates, nstates)."""
        ...

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Return the optional second state derivatives of ``F_Omega``.

        ``StateDerivatives.linear`` for an interior linear in u;
        ``StateDerivatives.none()`` when the curvature is not supplied.
        """
        ...

    def interior_is_time_invariant(self) -> bool:
        """Whether ``F_Omega(u, t)`` is DECLARED independent of ``t``:
        every coefficient and forcing it holds is declared
        time-independent."""
        ...


@runtime_checkable
class GalerkinPhysicsProtocol(
    GalerkinInteriorOperatorProtocol[Array], Protocol[Array]
):
    """A Galerkin physics: the interior operator ``F_Omega`` and the mass.

    The Galerkin formulation is ``M du/dt = F(u, t)``, where ``M`` comes
    from the weak form of the time derivative.
    """

    def mass_matrix(self) -> Array:
        """Return the mass matrix, e.g. ``M_ij = integral(phi_i phi_j)``.

        Shape: (nstates, nstates).
        """
        ...
