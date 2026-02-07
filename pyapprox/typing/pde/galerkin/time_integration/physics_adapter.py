"""Adapter to use Galerkin physics with time integration from typing.pde.time.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

This adapter returns raw (unmodified) quantities:
  f(y, t) = spatial_residual(y, t)   (no Dirichlet row zeroing)
  jacobian = spatial_jacobian(y, t)  (no Dirichlet row replacement)
  mass_matrix = M                    (raw FEM mass matrix)
  apply_mass_matrix(v) = M @ v

Dirichlet BCs are enforced by ConstrainedTimeStepResidual, which wraps
the stepper and applies R[d] = y[d] - g(t), J[d,:] = e_d after the
stepper assembles the full Newton system.
"""

from typing import Generic, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
    GalerkinPhysicsWithParamJacobianProtocol,
    GalerkinPhysicsWithHVPProtocol,
)


class GalerkinPhysicsODEAdapter(Generic[Array]):
    """Adapter from GalerkinPhysics to ODEResidualProtocol.

    Returns raw M, F, J_F — no BC modifications:
    - f(y) = spatial_residual(y, t) (unmodified)
    - jacobian(y) = spatial_jacobian(y, t) (unmodified)
    - mass_matrix = M (raw FEM mass matrix)
    - apply_mass_matrix(v) = M @ v

    Dirichlet BCs are applied externally by ConstrainedTimeStepResidual.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt. Must have spatial_residual(),
        spatial_jacobian(), and dirichlet_dof_info() methods.

    Examples
    --------
    >>> ode_residual = GalerkinPhysicsODEAdapter(physics)
    >>> time_stepper = BackwardEulerResidual(ode_residual)
    """

    def __init__(self, physics: GalerkinPhysicsProtocol[Array]):
        if not hasattr(physics, "spatial_residual"):
            raise TypeError(
                f"{type(physics).__name__} does not have spatial_residual(). "
                "The implicit adapter requires this method."
            )
        if not hasattr(physics, "spatial_jacobian"):
            raise TypeError(
                f"{type(physics).__name__} does not have spatial_jacobian(). "
                "The implicit adapter requires this method."
            )
        if not hasattr(physics, "dirichlet_dof_info"):
            raise TypeError(
                f"{type(physics).__name__} does not have dirichlet_dof_info(). "
                "The implicit adapter requires this method."
            )

        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0

        # Cache raw mass matrix
        self._mass_cached = physics.mass_matrix()

        # Setup optional methods based on physics capabilities
        self._setup_optional_methods()

    def _setup_optional_methods(self) -> None:
        """Conditionally expose methods based on physics capabilities."""
        # Check for parameter Jacobian support
        if isinstance(self._physics, GalerkinPhysicsWithParamJacobianProtocol):
            self.nparams = self._nparams
            self.set_param = self._set_param
            self.param_jacobian = self._param_jacobian
            self.initial_param_jacobian = self._initial_param_jacobian

        # Check for HVP support
        if isinstance(self._physics, GalerkinPhysicsWithHVPProtocol):
            self.state_state_hvp = self._state_state_hvp
            self.state_param_hvp = self._state_param_hvp
            self.param_state_hvp = self._param_state_hvp
            self.param_param_hvp = self._param_param_hvp

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
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
        """Evaluate spatial residual F(y, t) (unmodified).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        return self._physics.spatial_residual(state, self._time)

    def jacobian(self, state: Array) -> Array:
        """Compute spatial Jacobian dF/du (unmodified).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        return self._physics.spatial_jacobian(state, self._time)

    def mass_matrix(self, nstates: int) -> Array:
        """Return the raw FEM mass matrix.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Mass matrix M. Shape: (nstates, nstates)
        """
        return self._mass_cached

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector: M @ vec.

        Parameters
        ----------
        vec : Array
            Vector to multiply. Shape: (nstates,)

        Returns
        -------
        Array
            M @ vec. Shape: (nstates,)
        """
        return self._bkd.dot(self._mass_cached, vec)

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and values at given time.

        Parameters
        ----------
        time : float
            Time at which to evaluate Dirichlet BCs.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices : Array
                Global DOF indices. Shape: (ndirichlet,)
            dof_values : Array
                Exact Dirichlet values. Shape: (ndirichlet,)
        """
        return self._physics.dirichlet_dof_info(time)

    # =========================================================================
    # Parameter Jacobian Methods (optional)
    # =========================================================================

    def _nparams(self) -> int:
        """Get the number of parameters."""
        return self._physics.nparams()

    def _set_param(self, param: Array) -> None:
        """Set the parameter values."""
        self._physics.set_param(param)

    def _param_jacobian(self, state: Array) -> Array:
        """Compute the parameter Jacobian dF/dp."""
        return self._physics.param_jacobian(state, self._time)

    def _initial_param_jacobian(self) -> Array:
        """Get Jacobian of initial condition w.r.t. parameters."""
        return self._physics.initial_param_jacobian()

    # =========================================================================
    # HVP Methods (optional)
    # =========================================================================

    def _state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute (d^2F/du^2)w contracted with adjoint."""
        return self._physics.state_state_hvp(state, self._time, adj_state, wvec)

    def _state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute (d^2F/dudp)v contracted with adjoint."""
        return self._physics.state_param_hvp(state, self._time, adj_state, vvec)

    def _param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute (d^2F/dpdu)w contracted with adjoint."""
        return self._physics.param_state_hvp(state, self._time, adj_state, wvec)

    def _param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute (d^2F/dp^2)v contracted with adjoint."""
        return self._physics.param_param_hvp(state, self._time, adj_state, vvec)

    def __repr__(self) -> str:
        return (
            f"GalerkinPhysicsODEAdapter(\n"
            f"  physics={self._physics!r},\n"
            f"  time={self._time},\n"
            f")"
        )
