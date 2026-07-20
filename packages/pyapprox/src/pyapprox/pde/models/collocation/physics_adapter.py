"""Parameterized adapter tiers bridging Physics to ODEResidualProtocol.

Capability is decided ONCE at construction by
:func:`create_collocation_physics_ode_residual`, which None-checks the
parameterization's :class:`ParamDerivatives` bundle to select a
fixed-tier adapter class (never ``hasattr``):

- :class:`CollocationPhysicsToODEResidualAdapter` (base tier, lives in
  ``pyapprox.pde.collocation.time_integration.physics_adapter``) — raw
  f/J/M, no parameters.
- :class:`CollocationPhysicsToODEResidualWithSetParamAdapter` — adds
  ``nparams``/``set_param`` (parameterization without derivative
  capability).
- :class:`CollocationPhysicsToODEResidualWithParamJacobianAdapter` — adds
  ``param_jacobian``/``initial_param_jacobian``/
  ``bc_flux_param_sensitivity`` (first-order bundle).
- :class:`CollocationPhysicsToODEResidualWithHVPAdapter` — adds the three
  parameterization HVPs (second-order bundle) and ``state_state_hvp``
  (from a physics satisfying ``PhysicsWithStateStateHVPProtocol``).

Each tier class declares its methods unconditionally, so downstream
capability discovery (the ode stepper stack) sees real methods.
"""

from typing import Optional, overload

from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.protocols.physics import (
    PhysicsWithStateStateHVPProtocol,
)
from pyapprox.pde.collocation.time_integration.physics_adapter import (
    CollocationPhysicsToODEResidualAdapter,
)
from pyapprox.pde.parameterizations.derivatives import (
    BCFluxParamSensitivityFn,
    InitialParamJacobianFn,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class CollocationPhysicsToODEResidualWithSetParamAdapter(
    CollocationPhysicsToODEResidualAdapter[Array]
):
    """Adapter with a parameterization (evaluation-only tier).

    Adds ``nparams``/``set_param`` on top of the base tier. Selected by
    the factory when the parameterization's bundle declares no derivative
    capability.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
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
        super().__init__(physics, bkd)
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


class CollocationPhysicsToODEResidualWithParamJacobianAdapter(
    CollocationPhysicsToODEResidualWithSetParamAdapter[Array]
):
    """Adapter with first-order parameter derivatives.

    Adds ``param_jacobian``/``initial_param_jacobian``/
    ``bc_flux_param_sensitivity`` on top of the evaluation tier. Selected
    by the factory when the bundle has both parameter jacobians. The
    bundle is narrowed ONCE here into always-present private attributes.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        super().__init__(physics, bkd, parameterization)
        derivs = parameterization.param_derivatives()
        param_jacobian = derivs.param_jacobian
        initial_param_jacobian = derivs.initial_param_jacobian
        if param_jacobian is None or initial_param_jacobian is None:
            raise TypeError(
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has param_jacobian and initial_param_jacobian; "
                "use create_collocation_physics_ode_residual to select the right tier"
            )
        self._param_jacobian_fn: ParamJacobianFn[Array] = param_jacobian
        self._initial_param_jacobian_fn: InitialParamJacobianFn[Array] = (
            initial_param_jacobian
        )
        # Optional within this tier: absent capability returns None to
        # the (BC-wrapper) consumer, which skips the correction.
        self._bc_flux_fn: Optional[BCFluxParamSensitivityFn[Array]] = (
            derivs.bc_flux_param_sensitivity
        )

    def param_jacobian(self, state: Array) -> Array:
        """Compute the parameter Jacobian df/dp.

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

    def bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Optional[Array]:
        """Compute d(flux·n)/dp at boundary nodes, or None.

        Returns None when the parameterization does not provide the
        capability (the BC wrapper then skips the correction).
        """
        if self._bc_flux_fn is None or self._current_params_1d is None:
            return None
        return self._bc_flux_fn(
            state,
            time,
            self._current_params_1d,
            bc_indices,
            normals,
        )


class CollocationPhysicsToODEResidualWithHVPAdapter(
    CollocationPhysicsToODEResidualWithParamJacobianAdapter[Array]
):
    """Adapter with second-order parameter derivatives.

    Adds the three parameterization HVP contractions (from the bundle)
    and ``state_state_hvp`` (from the physics) on top of the sensitivity
    tier. Selected by the factory when the bundle has all three HVPs and
    the physics satisfies ``PhysicsWithStateStateHVPProtocol``.
    """

    def __init__(
        self,
        physics: PhysicsWithStateStateHVPProtocol[Array],
        bkd: Backend[Array],
        parameterization: ParameterizationProtocol[Array],
    ) -> None:
        if not isinstance(physics, PhysicsWithStateStateHVPProtocol):
            raise TypeError(
                f"{type(self).__name__} requires a physics with "
                f"state_state_hvp, got {type(physics).__name__}"
            )
        super().__init__(physics, bkd, parameterization)
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
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has all three HVP contractions; use "
                "create_collocation_physics_ode_residual to select the right tier"
            )
        self._hvp_physics = physics
        self._param_param_hvp_fn: ParamHVPFn[Array] = param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = param_state_hvp

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy^2) w. Shape: (nstates,)."""
        return self._hvp_physics.state_state_hvp(
            state, adj_state, wvec, self._time
        )

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp^2) v. Shape: (nparams,)."""
        return self._param_param_hvp_fn(
            state, self._time, self._require_params(), adj_state, vvec
        )

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dy dp) v. Shape: (nstates,)."""
        return self._state_param_hvp_fn(
            state, self._time, self._require_params(), adj_state, vvec
        )

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2f/dp dy) w. Shape: (nparams,)."""
        return self._param_state_hvp_fn(
            state, self._time, self._require_params(), adj_state, wvec
        )


@overload
def create_collocation_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: None = None,
) -> CollocationPhysicsToODEResidualAdapter[Array]: ...


@overload
def create_collocation_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: ParameterizationProtocol[Array],
) -> CollocationPhysicsToODEResidualWithSetParamAdapter[Array]: ...


def create_collocation_physics_ode_residual(
    physics: PhysicsProtocol[Array],
    bkd: Backend[Array],
    parameterization: Optional[ParameterizationProtocol[Array]] = None,
) -> CollocationPhysicsToODEResidualAdapter[Array]:
    """Create the widest adapter tier the inputs support.

    Capability enters the stepper stack exactly here: the factory
    None-checks the parameterization's ParamDerivatives bundle (and
    isinstance-checks the physics for ``state_state_hvp``) once, then
    everything above sees unconditional fixed-tier methods.

    Parameters
    ----------
    physics : PhysicsProtocol
        The collocation physics object to adapt.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol, optional
        Maps parameter vectors to physics coefficients.

    Returns
    -------
    CollocationPhysicsToODEResidualAdapter
        The widest tier supported by the bundle and physics.
    """
    if parameterization is None:
        return CollocationPhysicsToODEResidualAdapter(physics, bkd)
    if not isinstance(parameterization, ParameterizationProtocol):
        raise TypeError(
            f"parameterization must satisfy ParameterizationProtocol, "
            f"got {type(parameterization).__name__}"
        )
    derivs = parameterization.param_derivatives()
    has_first_order = (
        derivs.param_jacobian is not None
        and derivs.initial_param_jacobian is not None
    )
    has_hvps = (
        derivs.param_param_hvp is not None
        and derivs.state_param_hvp is not None
        and derivs.param_state_hvp is not None
    )
    if (
        has_first_order
        and has_hvps
        and isinstance(physics, PhysicsWithStateStateHVPProtocol)
    ):
        return CollocationPhysicsToODEResidualWithHVPAdapter(
            physics, bkd, parameterization
        )
    if has_first_order:
        return CollocationPhysicsToODEResidualWithParamJacobianAdapter(
            physics, bkd, parameterization
        )
    return CollocationPhysicsToODEResidualWithSetParamAdapter(
        physics, bkd, parameterization
    )
