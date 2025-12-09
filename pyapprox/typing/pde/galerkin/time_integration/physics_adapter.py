"""Adapter to use Galerkin physics with time integration from typing.pde.time.

The time module expects ODEResidualProtocol: dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

This adapter translates between the two interfaces, properly handling
the mass matrix in the time integration.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
    GalerkinPhysicsWithParamJacobianProtocol,
    GalerkinPhysicsWithHVPProtocol,
)


class GalerkinPhysicsODEAdapter(Generic[Array]):
    """Adapter from GalerkinPhysics to ODEResidualProtocol.

    The time steppers in typing.pde.time expect the standard ODE form:
        du/dt = f(u, t)

    Galerkin physics provides the weak form:
        M * du/dt = F(u, t)

    This adapter transforms F to f by solving: f = M^{-1} * F
    Similarly, the Jacobian becomes: df/du = M^{-1} * dF/du

    The mass matrix method returns the identity since the transformation
    absorbs M into the residual/jacobian.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt.

    Examples
    --------
    >>> from pyapprox.typing.pde.galerkin import (
    ...     StructuredMesh1D, LagrangeBasis, LinearAdvectionDiffusionReaction
    ... )
    >>> from pyapprox.typing.pde.galerkin.time_integration import (
    ...     GalerkinPhysicsODEAdapter
    ... )
    >>> from pyapprox.typing.pde.time.implicit_steppers import BackwardEulerResidual
    >>> # Setup physics
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> physics = LinearAdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, bkd=bkd
    ... )
    >>> # Create adapter and time stepper
    >>> ode_residual = GalerkinPhysicsODEAdapter(physics)
    >>> time_stepper = BackwardEulerResidual(ode_residual)
    """

    def __init__(self, physics: GalerkinPhysicsProtocol[Array]):
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0

        # Cache the mass matrix and its factorization for efficiency
        self._mass_matrix_cached: Optional[Array] = None
        self._identity_cached: Optional[Array] = None

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

    def _get_mass_matrix(self) -> Array:
        """Get the cached FEM mass matrix."""
        if self._mass_matrix_cached is None:
            self._mass_matrix_cached = self._physics.mass_matrix()
        return self._mass_matrix_cached

    def __call__(self, state: Array) -> Array:
        """Evaluate the ODE residual f(u, t) = M^{-1} * F(u, t).

        For Galerkin with M * du/dt = F(u, t), we transform to standard form
        du/dt = f(u, t) by solving f = M^{-1} * F.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Residual f(u, t) = M^{-1} * F(u, t). Shape: (nstates,)
        """
        F = self._physics.residual(state, self._time)
        M = self._get_mass_matrix()
        return self._bkd.solve(M, F)

    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian df/du = M^{-1} * dF/du.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian df/du. Shape: (nstates, nstates)
        """
        J_F = self._physics.jacobian(state, self._time)
        M = self._get_mass_matrix()
        # Solve M * J = J_F for each column of J
        return self._bkd.solve(M, J_F)

    def mass_matrix(self, nstates: int) -> Array:
        """Return the identity matrix.

        Since we transform the Galerkin system M * du/dt = F to standard form
        du/dt = f = M^{-1} * F, the effective mass matrix is now identity.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Identity matrix. Shape: (nstates, nstates)
        """
        if self._identity_cached is None:
            self._identity_cached = self._bkd.eye(nstates)
        return self._identity_cached

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
