"""Derivatives of the constrained Galerkin residual, from the raw ones.

Parameterizations and physics supply derivatives of the RAW spatial
residual ``F``. Every consumer differentiates the CONSTRAINED residual,
whose essential rows are ``y_d - g(t)`` (steady, and the end of each
time step) or carry the parameter-independent boundary velocity
``g_dot(t)`` (the neutralized ODE). Either way those rows do not depend
on the parameters, and are linear in the state, so:

- the parameter Jacobian has its essential rows zeroed;
- every second-derivative contraction ``lambda^T (d^2 F) ...`` receives
  the adjoint (or weight) with its essential entries zeroed. State-shaped
  outputs are NOT masked: their entries at essential indices are genuine.

This module is the one place that correction is made, for the steady
state equation and the transient adapters alike. The constraint set is
Galerkin's (essential rows are the replaced rows).
"""

from typing import Generic

from pyapprox.ode.state_derivatives import StateDerivatives, StateStateHVPFn
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamHVPFn,
    ParamDerivatives,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.util.backends.protocols import Array


class _RowZeroedParamJacobian(Generic[Array]):
    """``dF/dp`` with essential rows zeroed."""

    def __init__(
        self,
        raw: ParamJacobianFn[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> None:
        self._raw = raw
        self._constraint_set = constraint_set

    def __call__(self, state: Array, time: float, params_1d: Array) -> Array:
        return self._constraint_set.zero_rows(
            self._raw(state, time, params_1d)
        )


class _WeightZeroedParamHVP(Generic[Array]):
    """A parameter-facing contraction with the adjoint's essential
    entries zeroed."""

    def __init__(
        self,
        raw: ParamHVPFn[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> None:
        self._raw = raw
        self._constraint_set = constraint_set

    def __call__(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vec: Array,
    ) -> Array:
        return self._raw(
            state,
            time,
            params_1d,
            self._constraint_set.zero_entries(adj_state),
            vec,
        )


class _WeightZeroedInitialParamHVP(Generic[Array]):
    """The initial-state curvature with the weight's essential entries
    zeroed."""

    def __init__(
        self,
        raw: InitialParamHVPFn[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> None:
        self._raw = raw
        self._constraint_set = constraint_set

    def __call__(self, params_1d: Array, weight: Array, vvec: Array) -> Array:
        return self._raw(
            params_1d, self._constraint_set.zero_entries(weight), vvec
        )


class _WeightZeroedStateStateHVP(Generic[Array]):
    """``lambda^T (d^2F/du^2) w`` with the adjoint's essential entries
    zeroed."""

    def __init__(
        self,
        raw: StateStateHVPFn[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> None:
        self._raw = raw
        self._constraint_set = constraint_set

    def __call__(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        return self._raw(
            state, self._constraint_set.zero_entries(adj_state), wvec, time
        )


def constrain_param_derivatives(
    derivatives: ParamDerivatives[Array],
    constraint_set: ConstraintSetProtocol[Array],
) -> ParamDerivatives[Array]:
    """Return the parameter derivatives of the constrained residual.

    Absent fields stay absent. ``initial_param_jacobian`` and
    ``bc_flux_param_sensitivity`` pass through unchanged: the initial
    state's essential rows are set where the initial state is built, and
    the flux sensitivity is not a residual row.
    """
    param_jacobian = derivatives.param_jacobian
    initial_param_hvp = derivatives.initial_param_hvp
    param_param_hvp = derivatives.param_param_hvp
    state_param_hvp = derivatives.state_param_hvp
    param_state_hvp = derivatives.param_state_hvp
    return ParamDerivatives(
        param_jacobian=(
            None
            if param_jacobian is None
            else _RowZeroedParamJacobian(param_jacobian, constraint_set)
        ),
        initial_param_jacobian=derivatives.initial_param_jacobian,
        initial_param_hvp=(
            None
            if initial_param_hvp is None
            else _WeightZeroedInitialParamHVP(
                initial_param_hvp, constraint_set
            )
        ),
        param_param_hvp=(
            None
            if param_param_hvp is None
            else _WeightZeroedParamHVP(param_param_hvp, constraint_set)
        ),
        state_param_hvp=(
            None
            if state_param_hvp is None
            else _WeightZeroedParamHVP(state_param_hvp, constraint_set)
        ),
        param_state_hvp=(
            None
            if param_state_hvp is None
            else _WeightZeroedParamHVP(param_state_hvp, constraint_set)
        ),
        bc_flux_param_sensitivity=derivatives.bc_flux_param_sensitivity,
    )


def constrain_state_state_hvp(
    state_state_hvp: StateStateHVPFn[Array],
    constraint_set: ConstraintSetProtocol[Array],
) -> StateStateHVPFn[Array]:
    """Return the state curvature of the constrained residual."""
    return _WeightZeroedStateStateHVP(state_state_hvp, constraint_set)


def constrain_state_derivatives(
    derivatives: StateDerivatives[Array],
    constraint_set: ConstraintSetProtocol[Array],
) -> StateDerivatives[Array]:
    """Return the state derivatives of the constrained residual; absent
    stays absent."""
    state_state_hvp = derivatives.state_state_hvp
    if state_state_hvp is None:
        return StateDerivatives.none()
    return StateDerivatives.second_order(
        constrain_state_state_hvp(state_state_hvp, constraint_set)
    )
