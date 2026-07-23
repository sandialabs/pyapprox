"""Parameterized adapter tiers for Galerkin physics time integration.

Capability is decided ONCE at construction by
:func:`create_galerkin_physics_ode_residual`, which None-checks the
parameterization's :class:`ParamDerivatives` bundle to select a
fixed-tier adapter class (never ``hasattr``):

- :class:`GalerkinPhysicsToODEResidualAdapter` (base tier, lives in
  ``pyapprox.pde.galerkin.time_integration.physics_adapter``) — raw
  f/J/M, no parameters.
- :class:`GalerkinPhysicsToODEResidualWithSetParamAdapter` — adds
  ``nparams``/``set_param`` (parameterization without derivative
  capability).
- :class:`GalerkinPhysicsToODEResidualWithParamJacobianAdapter` — adds
  ``param_jacobian``/``initial_param_jacobian`` (first-order bundle).
- :class:`GalerkinPhysicsToODEResidualWithHVPAdapter` — adds the three
  parameterization HVP contractions (second-order bundle) and
  ``state_state_hvp`` (from the physics).
"""

from typing import Optional, overload

from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
    GalerkinPhysicsWithStateStateHVPProtocol,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamJacobianFn,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array


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
        if parameterization.physics() is not physics:
            raise ValueError(
                "parameterization binds a different physics instance "
                "than the one passed to the adapter; construct one "
                "parameterization per physics"
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
        self._parameterization.apply(param)

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
            state, self._time, self._require_params()
        )

    def initial_param_jacobian(self) -> Array:
        """Compute d(initial_state)/d(params).

        Returns
        -------
        Array
            Initial-condition Jacobian. Shape: (nstates, nparams)
        """
        return self._initial_param_jacobian_fn(self._require_params())


class GalerkinPhysicsToODEResidualWithHVPAdapter(
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter[Array]
):
    """Adapter with second-order parameter derivatives.

    Adds the three parameterization HVP contractions (from the bundle)
    and ``state_state_hvp`` (from the physics) on top of the
    first-order tier. Selected by the factory when the bundle has all
    three HVPs and the physics satisfies
    ``GalerkinPhysicsWithStateStateHVPProtocol``. All contractions are
    RAW (no Dirichlet handling) — the BC-enforcing wrapper owns that.
    """

    def __init__(
        self,
        physics: GalerkinPhysicsWithStateStateHVPProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(
            physics, GalerkinPhysicsWithStateStateHVPProtocol
        ):
            raise TypeError(
                f"{type(self).__name__} requires a physics with "
                f"state_state_hvp, got {type(physics).__name__}"
            )
        super().__init__(physics, parameterization)
        derivs = parameterization.param_derivatives()
        param_param_hvp = derivs.param_param_hvp
        state_param_hvp = derivs.state_param_hvp
        param_state_hvp = derivs.param_state_hvp
        if (
            param_param_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
        ):
            raise TypeError(
                f"{type(self).__name__} requires a parameterization "
                "whose bundle has all three HVP contractions; use "
                "create_galerkin_physics_ode_residual to select the "
                "right tier"
            )
        self._hvp_physics = physics
        self._param_param_hvp_fn: ParamHVPFn[Array] = param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = param_state_hvp

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dy^2) w. Shape: (nstates,)."""
        return self._hvp_physics.state_state_hvp(
            state, adj_state, wvec, self._time
        )

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dp^2) v. Shape: (nparams,)."""
        return self._param_param_hvp_fn(
            state, self._time, self._require_params(), adj_state, vvec
        )

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dy dp) v. Shape: (nstates,)."""
        return self._state_param_hvp_fn(
            state, self._time, self._require_params(), adj_state, vvec
        )

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dp dy) w. Shape: (nparams,)."""
        return self._param_state_hvp_fn(
            state, self._time, self._require_params(), adj_state, wvec
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
    None-checks the parameterization's ParamDerivatives bundle (and
    isinstance-checks the physics for ``state_state_hvp``) once, then
    everything above sees unconditional fixed-tier methods.

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
        if (
            derivs.param_param_hvp is not None
            and derivs.state_param_hvp is not None
            and derivs.param_state_hvp is not None
            and isinstance(
                physics, GalerkinPhysicsWithStateStateHVPProtocol
            )
        ):
            return GalerkinPhysicsToODEResidualWithHVPAdapter(
                physics, parameterization
            )
        return GalerkinPhysicsToODEResidualWithParamJacobianAdapter(
            physics, parameterization
        )
    return GalerkinPhysicsToODEResidualWithSetParamAdapter(
        physics, parameterization
    )
