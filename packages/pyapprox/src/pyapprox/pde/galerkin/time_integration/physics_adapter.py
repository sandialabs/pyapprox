"""Adapter to use Galerkin physics with time integration from pyapprox.ode.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

The adapters return raw (unmodified) quantities:
  f(y, t) = spatial_residual(y, t)   (no Dirichlet row zeroing)
  jacobian = spatial_jacobian(y, t)  (no Dirichlet row replacement)
  mass_matrix() = MassMatrixProtocol wrapping raw FEM mass matrix

Dirichlet BCs are enforced by the BC-enforcing time residual wrapper,
which wraps the stepper and applies R[d] = y[d] - g(t), J[d,:] = e_d
after the stepper assembles the full Newton system.

Capability is decided ONCE at construction by
:func:`create_galerkin_physics_ode_residual`, which None-checks the
parameterization's :class:`ParamDerivatives` bundle to select a
fixed-tier adapter class (never ``hasattr``):

- :class:`GalerkinPhysicsToODEResidualAdapter` — raw f/J/M, no
  parameters.
- :class:`GalerkinPhysicsToODEResidualWithSetParamAdapter` — adds
  ``nparams``/``set_param`` (parameterization without derivative
  capability).
- :class:`GalerkinPhysicsToODEResidualWithParamJacobianAdapter` — adds
  ``param_jacobian``/``initial_param_jacobian`` (first-order bundle).

No HVP tier exists yet: the galerkin parameterization HVP
implementations arrive in a later phase of the time-integration
refactor, so the factory caps at the WithParamJacobian tier even for
second-order bundles.
"""

from typing import Generic, Optional, Tuple, overload

from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
    ParameterizationProtocol,
)
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamJacobianFn,
    ParamJacobianFn,
)
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from GalerkinPhysics to ODEResidualProtocol (base tier).

    Returns raw M, F, J_F -- no BC modifications:
    - f(y) = spatial_residual(y, t) (unmodified)
    - jacobian(y) = spatial_jacobian(y, t) (unmodified)
    - mass_matrix() = MassMatrixProtocol wrapping M

    Dirichlet BCs are applied externally by the BC-enforcing time
    residual wrapper.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt. Must have spatial_residual(),
        spatial_jacobian(), and dirichlet_dof_info() methods.

    Examples
    --------
    >>> ode_residual = create_galerkin_physics_ode_residual(physics)
    >>> time_stepper = BackwardEulerHVP(ode_residual)
    """

    def __init__(self, physics: GalerkinPhysicsProtocol[Array]) -> None:
        if not isinstance(physics, GalerkinPhysicsProtocol):
            raise TypeError(
                f"physics must satisfy GalerkinPhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        # Cache mass matrix as value-object (handles sparse via splu)
        self._mass = create_mass_matrix(physics.mass_matrix(), self._bkd)

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

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the FEM mass matrix as a value-object."""
        return self._mass

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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"physics={type(self._physics).__name__})"
        )


class GalerkinPhysicsToODEResidualWithSetParamAdapter(
    GalerkinPhysicsToODEResidualAdapter[Array]
):
    """Adapter with a parameterization (evaluation-only tier).

    Adds ``nparams``/``set_param`` on top of the base tier. Selected by
    the factory when the parameterization's bundle declares no
    derivative capability.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt.
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients.
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        super().__init__(physics)
        self._parameterization = parameterization
        self._current_params_1d: Optional[Array] = None

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """Return the parameterization."""
        return self._parameterization

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def set_param(self, param: Array) -> None:
        """Set parameter values through the parameterization.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,) — the ODEResidual
            protocol convention (strictly validated; callers convert at
            the optimizer/ode boundary).
        """
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D with shape (nparams,), got shape "
                f"{tuple(param.shape)}"
            )
        self._current_params_1d = param
        self._parameterization.apply(self._physics, param)

    def _require_params(self) -> Array:
        """Return the current parameters or raise if set_param not called."""
        if self._current_params_1d is None:
            raise RuntimeError(
                "set_param() must be called before parameter derivatives"
            )
        return self._current_params_1d


class GalerkinPhysicsToODEResidualWithParamJacobianAdapter(
    GalerkinPhysicsToODEResidualWithSetParamAdapter[Array]
):
    """Adapter with first-order parameter derivatives.

    Adds ``param_jacobian``/``initial_param_jacobian`` on top of the
    evaluation tier. Selected by the factory when the bundle has both
    parameter jacobians. The bundle is narrowed ONCE here into
    always-present private attributes.
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        super().__init__(physics, parameterization)
        derivs = parameterization.param_derivatives()
        param_jacobian = derivs.param_jacobian
        initial_param_jacobian = derivs.initial_param_jacobian
        if param_jacobian is None or initial_param_jacobian is None:
            raise TypeError(
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has param_jacobian and initial_param_jacobian; use "
                "create_galerkin_physics_ode_residual to select the right "
                "tier"
            )
        self._param_jacobian_fn: ParamJacobianFn[Array] = param_jacobian
        self._initial_param_jacobian_fn: InitialParamJacobianFn[Array] = (
            initial_param_jacobian
        )

    def param_jacobian(self, state: Array) -> Array:
        """Compute the parameter Jacobian dF/dp (raw, no Dirichlet).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._param_jacobian_fn(
            self._physics, state, self._time, self._require_params()
        )

    def initial_param_jacobian(self) -> Array:
        """Compute d(initial_state)/d(params).

        Returns
        -------
        Array
            Initial-condition Jacobian. Shape: (nstates, nparams)
        """
        return self._initial_param_jacobian_fn(
            self._physics, self._require_params()
        )


@overload
def create_galerkin_physics_ode_residual(
    physics: GalerkinPhysicsProtocol[Array],
    parameterization: None = None,
) -> GalerkinPhysicsToODEResidualAdapter[Array]: ...


@overload
def create_galerkin_physics_ode_residual(
    physics: GalerkinPhysicsProtocol[Array],
    parameterization: ParameterizationProtocol[Array],
) -> GalerkinPhysicsToODEResidualWithSetParamAdapter[Array]: ...


def create_galerkin_physics_ode_residual(
    physics: GalerkinPhysicsProtocol[Array],
    parameterization: Optional[ParameterizationProtocol[Array]] = None,
) -> GalerkinPhysicsToODEResidualAdapter[Array]:
    """Create the widest adapter tier the inputs support.

    Capability enters the stepper stack exactly here: the factory
    None-checks the parameterization's ParamDerivatives bundle once,
    then everything above sees unconditional fixed-tier methods. No
    HVP tier exists yet (galerkin param-HVP implementations arrive in a
    later phase of the time-integration refactor), so second-order
    bundles also produce the WithParamJacobian tier.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt.
    parameterization : ParameterizationProtocol, optional
        Maps parameter vectors to physics coefficients.

    Returns
    -------
    GalerkinPhysicsToODEResidualAdapter
        The widest tier supported by the bundle.
    """
    if parameterization is None:
        return GalerkinPhysicsToODEResidualAdapter(physics)
    if not isinstance(parameterization, ParameterizationProtocol):
        raise TypeError(
            f"parameterization must satisfy ParameterizationProtocol, "
            f"got {type(parameterization).__name__}"
        )
    derivs = parameterization.param_derivatives()
    if (
        derivs.param_jacobian is not None
        and derivs.initial_param_jacobian is not None
    ):
        return GalerkinPhysicsToODEResidualWithParamJacobianAdapter(
            physics, parameterization
        )
    return GalerkinPhysicsToODEResidualWithSetParamAdapter(
        physics, parameterization
    )
