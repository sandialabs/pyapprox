"""Adapter to bridge Physics to ODEResidualProtocol.

This module provides an adapter that wraps a collocation Physics object
to conform to the ODEResidualProtocol used by time integrators.

Key interface differences:
- Physics: residual(state, time), jacobian(state, time), mass_matrix()
- ODEResidual: __call__(state), jacobian(state), set_time(time), mass_matrix(nstates)

The adapter uses dynamic binding to expose param_jacobian and HVP methods
only if the underlying physics supports them.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols import PhysicsProtocol


class PhysicsToODEResidualAdapter(Generic[Array]):
    """Adapter from Physics to ODEResidualProtocol.

    Wraps a collocation Physics object to provide the ODEResidualProtocol
    interface expected by time integrators.

    The adapter:
    - Stores time internally via set_time()
    - Translates __call__(state) to physics.residual(state, time)
    - Applies boundary conditions to residual and Jacobian
    - Dynamically exposes param_jacobian/HVP if physics supports them

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
    >>> ode_residual = PhysicsToODEResidualAdapter(physics, bkd)
    >>> ode_residual.set_time(0.0)
    >>> f_y = ode_residual(state)  # Calls physics.residual with BCs applied
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
    ):
        self._physics = physics
        self._bkd = bkd
        self._time = 0.0
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Expose optional methods based on physics capabilities.

        Uses dynamic binding to add param_jacobian, initial_param_jacobian,
        and HVP methods only if the underlying physics implements them.
        """
        # Check for parameter Jacobian support
        if hasattr(self._physics, "param_jacobian"):
            # Bind the wrapper methods
            self.nparams = self._physics.nparams
            self.set_param = self._physics.set_param
            self.param_jacobian = self._param_jacobian_impl
            self.initial_param_jacobian = self._physics.initial_param_jacobian

        # Check for HVP support
        if hasattr(self._physics, "state_state_hvp"):
            self.state_state_hvp = self._state_state_hvp_impl
            self.state_param_hvp = self._state_param_hvp_impl
            self.param_state_hvp = self._param_state_hvp_impl
            self.param_param_hvp = self._param_param_hvp_impl

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

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

        Computes the physics residual and applies boundary conditions.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Residual with boundary conditions applied. Shape: (nstates,)
        """
        residual = self._physics.residual(state, self._time)

        # Apply boundary conditions
        if self._physics.boundary_conditions():
            jacobian = self._physics.jacobian(state, self._time)
            residual, _ = self._physics.apply_boundary_conditions(
                residual, jacobian, state, self._time
            )

        return residual

    def jacobian(self, state: Array) -> Array:
        """Compute the state Jacobian df/dy.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian with boundary conditions applied. Shape: (nstates, nstates)
        """
        jacobian = self._physics.jacobian(state, self._time)

        # Apply boundary conditions
        if self._physics.boundary_conditions():
            residual = self._physics.residual(state, self._time)
            _, jacobian = self._physics.apply_boundary_conditions(
                residual, jacobian, state, self._time
            )

        return jacobian

    def mass_matrix(self, nstates: int) -> Array:
        """Return the mass matrix.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Parameters
        ----------
        nstates : int
            Number of states (unused, present for protocol compatibility).

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        return self._physics.mass_matrix()

    # =========================================================================
    # Optional methods (dynamically bound if physics supports them)
    # =========================================================================

    def _param_jacobian_impl(self, state: Array) -> Array:
        """Compute parameter Jacobian df/dp.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._physics.param_jacobian(state, self._time)

    def _state_state_hvp_impl(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy^2) w.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        wvec : Array
            Direction vector. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        return self._physics.state_state_hvp(state, adj_state, wvec, self._time)

    def _state_param_hvp_impl(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy dp) v.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        vvec : Array
            Parameter direction. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        return self._physics.state_param_hvp(state, adj_state, vvec, self._time)

    def _param_state_hvp_impl(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp dy) w.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        wvec : Array
            State direction. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        return self._physics.param_state_hvp(state, adj_state, wvec, self._time)

    def _param_param_hvp_impl(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp^2) v.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        vvec : Array
            Parameter direction. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        return self._physics.param_param_hvp(state, adj_state, vvec, self._time)

    def __repr__(self) -> str:
        has_param_jac = hasattr(self, "param_jacobian")
        has_hvp = hasattr(self, "state_state_hvp")
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"has_param_jacobian={has_param_jac}, "
            f"has_hvp={has_hvp})"
        )
