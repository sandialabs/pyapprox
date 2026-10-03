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

from dataclasses import dataclass
from typing import Callable, Generic, Optional

from pyapprox.ode.state_derivatives import StateDerivatives, StateStateHVPFn
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamHVPFn,
    ParamDerivatives,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.steady_view import SteadyViewProtocol
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


SteadyParamJacobianFn = Callable[[Array, Array], Array]
SteadyParamHVPFn = Callable[[Array, Array, Array, Array], Array]
SteadyStateStateHVPFn = Callable[[Array, Array, Array], Array]


class _AtTimeParamJacobian(Generic[Array]):
    """``(state, params) -> dF/dp`` at a bound time."""

    def __init__(self, fn: ParamJacobianFn[Array], time: float) -> None:
        self._fn = fn
        self._time = time

    def __call__(self, state: Array, params_1d: Array) -> Array:
        return self._fn(state, self._time, params_1d)


class _AtTimeParamHVP(Generic[Array]):
    """``(state, params, adj, vec) -> contraction`` at a bound time."""

    def __init__(self, fn: ParamHVPFn[Array], time: float) -> None:
        self._fn = fn
        self._time = time

    def __call__(
        self, state: Array, params_1d: Array, adj_state: Array, vec: Array
    ) -> Array:
        return self._fn(state, self._time, params_1d, adj_state, vec)


class _AtTimeStateStateHVP(Generic[Array]):
    """``(state, adj, w) -> contraction`` at a bound time."""

    def __init__(self, fn: StateStateHVPFn[Array], time: float) -> None:
        self._fn = fn
        self._time = time

    def __call__(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        return self._fn(state, adj_state, wvec, self._time)


@dataclass(frozen=True)
class SteadyConstrainedDerivatives(Generic[Array]):
    """Derivatives of a steady view's constrained residual, time-free.

    The view's time is bound into every function, so a steady consumer
    never passes one. Absent is ``None``.
    """

    param_jacobian: Optional[SteadyParamJacobianFn[Array]] = None
    param_param_hvp: Optional[SteadyParamHVPFn[Array]] = None
    state_param_hvp: Optional[SteadyParamHVPFn[Array]] = None
    param_state_hvp: Optional[SteadyParamHVPFn[Array]] = None
    state_state_hvp: Optional[SteadyStateStateHVPFn[Array]] = None


def steady_constrained_derivatives(
    view: SteadyViewProtocol[Array],
    param_derivatives: ParamDerivatives[Array],
) -> SteadyConstrainedDerivatives[Array]:
    """Constrain the raw derivatives and bind the view's time into them.

    The one place a steady consumer's derivatives meet time: the view
    holds it, and it is curried here, once.
    """
    constraint_set = view.constraint_set()
    time = view.time()
    constrained = constrain_param_derivatives(
        param_derivatives, constraint_set
    )
    state_state_hvp = (
        view.spatial_operator().state_derivatives().state_state_hvp
    )

    def at_time(
        fn: Optional[ParamHVPFn[Array]],
    ) -> Optional[SteadyParamHVPFn[Array]]:
        return None if fn is None else _AtTimeParamHVP(fn, time)

    return SteadyConstrainedDerivatives(
        param_jacobian=(
            None
            if constrained.param_jacobian is None
            else _AtTimeParamJacobian(constrained.param_jacobian, time)
        ),
        param_param_hvp=at_time(constrained.param_param_hvp),
        state_param_hvp=at_time(constrained.state_param_hvp),
        param_state_hvp=at_time(constrained.param_state_hvp),
        state_state_hvp=(
            None
            if state_state_hvp is None
            else _AtTimeStateStateHVP(
                constrain_state_state_hvp(state_state_hvp, constraint_set),
                time,
            )
        ),
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
