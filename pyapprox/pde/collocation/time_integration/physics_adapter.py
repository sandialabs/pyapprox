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

from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


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
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        if parameterization is not None and not isinstance(
            parameterization, ParameterizationProtocol
        ):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._physics = physics
        self._bkd = bkd
        self._time = 0.0
        self._parameterization = parameterization
        self._current_params_1d: Optional[Array] = None
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """Expose optional methods based on physics/parameterization capabilities.

        Parameterization path takes priority over legacy physics path.
        Uses dynamic binding to add param_jacobian, initial_param_jacobian,
        and HVP methods only if the underlying source supports them.
        """
        if self._parameterization is not None:
            # Parameterization path
            self.nparams = self._parameterization.nparams
            self.set_param = self._set_param_via_parameterization
            if hasattr(self._parameterization, "param_jacobian"):
                self.param_jacobian = self._param_jacobian_via_parameterization
            if hasattr(self._parameterization, "initial_param_jacobian"):
                self.initial_param_jacobian = (
                    self._initial_param_jacobian_via_parameterization
                )
            if hasattr(self._parameterization, "param_param_hvp"):
                self.param_param_hvp = self._param_param_hvp_via_parameterization
            if hasattr(self._parameterization, "state_param_hvp"):
                self.state_param_hvp = self._state_param_hvp_via_parameterization
            if hasattr(self._parameterization, "param_state_hvp"):
                self.param_state_hvp = self._param_state_hvp_via_parameterization
            # state_state_hvp stays from physics (unchanged)
            if hasattr(self._physics, "state_state_hvp"):
                self.state_state_hvp = self._state_state_hvp_impl
        elif hasattr(self._physics, "param_jacobian"):
            # Legacy path (unchanged)
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

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector. Identity for standard collocation."""
        return self._physics.apply_mass_matrix(vec)

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

    # =========================================================================
    # Parameterization delegation methods
    # =========================================================================

    def _set_param_via_parameterization(self, param: Array) -> None:
        """Set parameter via parameterization."""
        self._current_params_1d = param
        self._parameterization.apply(self._physics, param)

    def _param_jacobian_via_parameterization(self, state: Array) -> Array:
        """Compute param Jacobian via parameterization."""
        return self._parameterization.param_jacobian(
            self._physics, state, self._time, self._current_params_1d
        )

    def _initial_param_jacobian_via_parameterization(self) -> Array:
        """Compute initial param Jacobian via parameterization."""
        return self._parameterization.initial_param_jacobian(
            self._physics, self._current_params_1d
        )

    def _param_param_hvp_via_parameterization(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute param-param HVP via parameterization."""
        return self._parameterization.param_param_hvp(
            self._physics, state, self._time, self._current_params_1d, adj_state, vvec
        )

    def _state_param_hvp_via_parameterization(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute state-param HVP via parameterization."""
        return self._parameterization.state_param_hvp(
            self._physics, state, self._time, self._current_params_1d, adj_state, vvec
        )

    def _param_state_hvp_via_parameterization(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute param-state HVP via parameterization."""
        return self._parameterization.param_state_hvp(
            self._physics, state, self._time, self._current_params_1d, adj_state, wvec
        )

    def bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> object:
        """Compute d(flux·n)/dp at boundary nodes, or None."""
        if (
            self._parameterization is not None
            and hasattr(self._parameterization, "bc_flux_param_sensitivity")
            and self._current_params_1d is not None
        ):
            return self._parameterization.bc_flux_param_sensitivity(
                self._physics,
                state,
                time,
                self._current_params_1d,
                bc_indices,
                normals,
            )
        return None

    def __repr__(self) -> str:
        has_param_jac = hasattr(self, "param_jacobian")
        has_hvp = hasattr(self, "state_state_hvp")
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"has_param_jacobian={has_param_jac}, "
            f"has_hvp={has_hvp})"
        )
