"""Adapter to bridge Physics to ODEResidualProtocol (base tier).

This module provides the non-parameterized base adapter that wraps a
collocation Physics object to conform to the ODEResidualProtocol used by
time integrators.

Key interface differences:
- Physics: residual(state, time), jacobian(state, time), mass_matrix()
- ODEResidual: __call__(state), jacobian(state), set_time(time), mass_matrix(nstates)

The parameterized adapter tiers and the capability-selecting factory
live in ``pyapprox.pde.models.collocation.physics_adapter`` — the
models layer owns everything that requires both a physics and a
parameterization.
"""

from typing import Generic

from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.util.backends.protocols import Array, Backend


class CollocationPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from Physics to ODEResidualProtocol (base tier).

    Wraps a collocation Physics object to provide the raw
    ODEResidualProtocol interface expected by time integrators:

    - Stores time internally via set_time()
    - Translates __call__(state) to physics.residual(state, time)

    Boundary conditions are NOT applied here; for transient problems they
    are applied to the Newton residual by the BC-enforcing time residual
    wrapper.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.

    Examples
    --------
    >>> physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
    >>> physics.set_boundary_conditions([bc_left, bc_right])
    >>> ode_residual = CollocationPhysicsToODEResidualAdapter(physics, bkd)
    >>> ode_residual.set_time(0.0)
    >>> f_y = ode_residual(state)
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"physics must satisfy PhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        self._time = 0.0
        self._mass = create_mass_matrix(physics.mass_matrix(), bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> PhysicsProtocol[Array]:
        """Return the wrapped physics object."""
        return self._physics

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate the ODE residual f(y, t).

        Returns the physics residual WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton residual by the time integrator, not to the physics residual.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Physics residual. Shape: (nstates,)
        """
        return self._physics.residual(state, self._time)

    def jacobian(self, state: Array) -> Array:
        """Compute the state Jacobian df/dy.

        Returns the physics Jacobian WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton Jacobian by the time integrator.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Physics Jacobian. Shape: (nstates, nstates)
        """
        return self._physics.jacobian(state, self._time)

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the mass matrix."""
        return self._mass

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"physics={type(self._physics).__name__})"
        )
