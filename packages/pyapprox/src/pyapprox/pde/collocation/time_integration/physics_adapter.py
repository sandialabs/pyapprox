"""Adapter to bridge Physics to ODEResidualProtocol (base tier).

This module provides the non-parameterized base adapter that wraps a
collocation Physics object to conform to the ODEResidualProtocol used by
time integrators.

Key interface differences:
- Physics: residual(state, time), jacobian(state, time), mass_matrix()
- ODEResidual: __call__(state, time), jacobian(state, time), mass_matrix()
  returning a mass value object

The parameterized adapter tiers and the capability-selecting factory
live in ``pyapprox.pde.models.collocation.physics_adapter`` — the
models layer owns everything that requires both a physics and a
parameterization.
"""

from typing import Generic, Optional

from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.residual_derivatives import (
    ODEResidualDerivatives,
    backend_state_space,
    dense_jacobian,
)
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendVectorSpace, DenseFormat


class CollocationPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from Physics to ODEResidualProtocol (base tier).

    Wraps a collocation Physics object to provide the raw
    ODEResidualProtocol interface expected by time integrators:

    - Translates __call__(state, time) to physics.residual(state, time)
    - Holds no time state

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
    >>> f_y = ode_residual(state, 0.0)
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"physics must satisfy PhysicsProtocol, got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        self._mass = create_mass_matrix(physics.mass_matrix(), bkd)
        self._derivatives: Optional[
            ODEResidualDerivatives[Array, DenseFormat[Array]]
        ] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        """The residual's derivatives as operators, built on first request
        (a parameterized tier needs its parameterization set first) and
        kept: here the dense state Jacobian only."""
        if self._derivatives is None:
            self._derivatives = self._build_derivatives()
        return self._derivatives

    def _state_space(self) -> BackendVectorSpace[Array]:
        """The state space on the backend stack."""
        return backend_state_space(self._bkd, self._mass.as_matrix().shape[0])

    def _build_derivatives(
        self,
    ) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        """This tier's bundle; each parameterized tier overrides it."""
        return ODEResidualDerivatives.state_only(
            dense_jacobian(self.jacobian, self._state_space(), self._bkd)
        )

    def physics(self) -> PhysicsProtocol[Array]:
        """Return the wrapped physics object."""
        return self._physics

    def __call__(self, state: Array, time: float) -> Array:
        """Evaluate the ODE residual f(y, t).

        Returns the physics residual WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton residual by the time integrator, not to the physics residual.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.

        Returns
        -------
        Array
            Physics residual. Shape: (nstates,)
        """
        return self._physics.residual(state, time)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute the state Jacobian df/dy.

        Returns the physics Jacobian WITHOUT boundary conditions applied.
        For transient problems, boundary conditions should be applied to the
        Newton Jacobian by the time integrator.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.

        Returns
        -------
        Array
            Physics Jacobian. Shape: (nstates, nstates)
        """
        return self._physics.jacobian(state, time)

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the mass matrix."""
        return self._mass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(physics={type(self._physics).__name__})"
