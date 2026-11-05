"""A Galerkin right-hand side's derivatives bundle from its parts' bundles.

A parameterization supplies its derivatives as functions of the state,
time and parameter vector (:class:`ParamDerivatives`), and the physics
its state curvature as a function of the state, adjoint and direction.
A right-hand side's bundle (:class:`ODEResidualDerivatives`) wants them
as operators read at the current parameters. :func:`residual_derivatives`
makes that conversion, for every system built over one
:class:`ParameterState`: the solved system, whose functions have their
essential rows treated, and the unconstrained one, whose functions are
raw, convert the same way and read the same parameters.

The parameter and initial-state Jacobians arrive as dense matrices and
become dense operators here, the one place they are converted.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    Jacobian,
    ODEResidualDerivatives,
    dense_operator,
    matrix_free_initial_state_curvature,
    matrix_free_second_derivative,
)
from pyapprox.ode.state_derivatives import StateStateHVPFn
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamHVPFn,
    InitialParamJacobianFn,
    ParamDerivatives,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol, VectorSpaceProtocol

F = TypeVar("F", bound=FormatProtocol)


class _ParamJacobianAt(Generic[Array]):
    """``(state, time) -> dF/dp`` at the current parameters."""

    def __init__(
        self, fn: ParamJacobianFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self, state: Array, time: float) -> Array:
        return self._fn(state, time, self._parameters.current())


class _InitialParamJacobianAt(Generic[Array]):
    """``() -> dy_0/dp`` at the current parameters."""

    def __init__(
        self, fn: InitialParamJacobianFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self) -> Array:
        return self._fn(self._parameters.current())


class _InitialParamHVPAt(Generic[Array]):
    """``(weight, vec) -> sum_i weight_i (d^2 y_0,i/dp^2) vec`` at the
    current parameters."""

    def __init__(
        self, fn: InitialParamHVPFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self, weight: Array, vec: Array) -> Array:
        return self._fn(self._parameters.current(), weight, vec)


class _ParamHVPAt(Generic[Array]):
    """``(state, time, adj, vec) -> contraction`` at the current
    parameters."""

    def __init__(
        self, fn: ParamHVPFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self, state: Array, time: float, adj: Array, vec: Array) -> Array:
        return self._fn(state, time, self._parameters.current(), adj, vec)


class _StateStateHVP(Generic[Array]):
    """``(state, time, adj, vec) -> adj^T (d^2F/du^2) vec``, in the order
    the bundle's second derivatives take their arguments."""

    def __init__(self, fn: StateStateHVPFn[Array]) -> None:
        self._fn = fn

    def __call__(self, state: Array, time: float, adj: Array, vec: Array) -> Array:
        return self._fn(state, adj, vec, time)


def residual_derivatives(
    jacobian: Jacobian[Array, F],
    param_derivatives: ParamDerivatives[Array],
    state_state_hvp: Optional[StateStateHVPFn[Array]],
    parameters: ParameterState[Array],
    states: VectorSpaceProtocol[Array],
    params: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
) -> ODEResidualDerivatives[Array, F]:
    r"""The bundle of a right-hand side with state Jacobian ``jacobian``
    and the given parameter and state derivatives, read at the current
    parameters of ``parameters``.

    Its order is decided here, once, by which fields are present: first
    order with ``param_jacobian`` and ``initial_param_jacobian``; second
    order when, besides, the three parameter contractions,
    ``initial_param_hvp`` and ``state_state_hvp`` are all present; the
    state Jacobian alone otherwise. A Jacobian that cannot see the state
    (:math:`f` affine) gives the stated zero state curvature.

    Parameters
    ----------
    jacobian : Jacobian
        The state Jacobian.
    param_derivatives : ParamDerivatives
        The parameterization's derivatives of :math:`f`.
    state_state_hvp : StateStateHVPFn, optional
        :math:`\lambda^\top (\partial^2 f/\partial y^2) w`.
    parameters : ParameterState
        The parameters every field is read at.
    states, params : VectorSpaceProtocol
        The state and parameter spaces.
    bkd : Backend
        The backend of the states.
    """
    param_jacobian = param_derivatives.param_jacobian
    initial_param_jacobian = param_derivatives.initial_param_jacobian
    if param_jacobian is None or initial_param_jacobian is None:
        return ODEResidualDerivatives.state_only(jacobian)
    param = dense_operator(
        _ParamJacobianAt(param_jacobian, parameters), params, states, bkd
    )
    initial = dense_operator(
        _InitialParamJacobianAt(initial_param_jacobian, parameters),
        params,
        states,
        bkd,
    )
    initial_param_hvp = param_derivatives.initial_param_hvp
    param_param_hvp = param_derivatives.param_param_hvp
    state_param_hvp = param_derivatives.state_param_hvp
    param_state_hvp = param_derivatives.param_state_hvp
    if (
        initial_param_hvp is None
        or param_param_hvp is None
        or state_param_hvp is None
        or param_state_hvp is None
        or state_state_hvp is None
    ):
        return ODEResidualDerivatives.first_order(
            states, params, jacobian=jacobian, param=param, initial_state=initial
        )
    param_param = _ParamHVPAt(param_param_hvp, parameters)
    state_state = _StateStateHVP(state_state_hvp)
    return ODEResidualDerivatives.second_order(
        states,
        params,
        jacobian=jacobian,
        param=param,
        initial_state=initial,
        initial_state_curvature=matrix_free_initial_state_curvature(
            _InitialParamHVPAt(initial_param_hvp, parameters), params, bkd
        ),
        # A Jacobian that cannot see the state means f is affine, with
        # zero state curvature, stated as such.
        state_state=(
            STATED_ZERO
            if jacobian.affine()
            else matrix_free_second_derivative(
                state_state, state_state, states, states, bkd, True
            )
        ),
        state_param=matrix_free_second_derivative(
            _ParamHVPAt(state_param_hvp, parameters),
            _ParamHVPAt(param_state_hvp, parameters),
            params,
            states,
            bkd,
        ),
        param_param=matrix_free_second_derivative(
            param_param, param_param, params, params, bkd, True
        ),
    )
