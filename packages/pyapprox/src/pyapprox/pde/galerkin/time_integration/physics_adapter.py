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

Parameter sensitivity is provided through an optional
``ParameterizationProtocol`` object, which maps parameter vectors to
physics coefficients and computes chain-rule Jacobians.
"""

from typing import Generic, Optional, Tuple

from scipy.sparse import issparse

from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsODEAdapter(Generic[Array]):
    """Adapter from GalerkinPhysics to ODEResidualProtocol.

    Returns raw M, F, J_F -- no BC modifications:
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
    parameterization : ParameterizationProtocol, optional
        Optional parameterization mapping parameter vectors to physics
        coefficients. When provided, ``nparams``, ``set_param``,
        ``param_jacobian``, and ``initial_param_jacobian`` methods are
        exposed (if the parameterization supports them).

    Examples
    --------
    >>> ode_residual = GalerkinPhysicsODEAdapter(physics)
    >>> time_stepper = BackwardEulerHVP(ode_residual)

    With parameterization:

    >>> param = GalerkinLameParameterization(...)
    >>> ode_residual = GalerkinPhysicsODEAdapter(physics, param)
    >>> ode_residual.set_param(param_vector)
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        if parameterization is not None:
            if not isinstance(parameterization, ParameterizationProtocol):
                raise TypeError(
                    f"parameterization must satisfy "
                    f"ParameterizationProtocol, "
                    f"got {type(parameterization).__name__}"
                )
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        self._parameterization = parameterization
        self._current_params_1d: Optional[Array] = None

        # Cache raw mass matrix
        self._mass_cached = physics.mass_matrix()

        # Setup optional methods based on parameterization capabilities
        self._setup_optional_methods()

    def _setup_optional_methods(self) -> None:
        """Conditionally expose methods based on parameterization."""
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
            # HVP methods from parameterization if available
            for method_name in (
                "param_param_hvp",
                "state_param_hvp",
                "param_state_hvp",
            ):
                if hasattr(self._parameterization, method_name):
                    hvp_impl = getattr(
                        self,
                        f"_{method_name}_via_parameterization",
                        None,
                    )
                    if hvp_impl is not None:
                        setattr(self, method_name, hvp_impl)
            # state_state_hvp stays from physics
            if hasattr(self._physics, "state_state_hvp"):
                self.state_state_hvp = self._state_state_hvp

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
        # Use @ with to_numpy because mass matrix may be sparse (scipy).
        # Cannot use bkd.dot() — it doesn't handle sparse matrices.
        if issparse(self._mass_cached):
            vec_np = self._bkd.to_numpy(vec)
            return self._bkd.asarray(self._mass_cached @ vec_np)
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
    # Parameterization delegation methods
    # =========================================================================

    def _set_param_via_parameterization(self, param: Array) -> None:
        """Set parameter values through parameterization."""
        self._current_params_1d = param
        self._parameterization.apply(self._physics, param)

    def _param_jacobian_via_parameterization(
        self,
        state: Array,
    ) -> Array:
        """Compute parameter Jacobian through parameterization."""
        if self._current_params_1d is None:
            raise RuntimeError("set_param() must be called before param_jacobian()")
        return self._parameterization.param_jacobian(
            self._physics, state, self._time, self._current_params_1d
        )

    def _initial_param_jacobian_via_parameterization(self) -> Array:
        """Get initial condition Jacobian through parameterization."""
        if self._current_params_1d is None:
            raise RuntimeError(
                "set_param() must be called before initial_param_jacobian()"
            )
        return self._parameterization.initial_param_jacobian(
            self._physics, self._current_params_1d
        )

    # =========================================================================
    # HVP Methods (optional, from physics or parameterization)
    # =========================================================================

    def _state_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        """Compute (d^2F/du^2)w contracted with adjoint."""
        return self._physics.state_state_hvp(state, adj_state, wvec, self._time)

    def __repr__(self) -> str:
        parts = [
            f"GalerkinPhysicsODEAdapter(\n"
            f"  physics={self._physics!r},\n"
            f"  time={self._time},\n"
        ]
        if self._parameterization is not None:
            parts.append(f"  parameterization={self._parameterization!r},\n")
        parts.append(")")
        return "".join(parts)
