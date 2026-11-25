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

The parameter Jacobian arrives as an action and becomes a matrix-free
operator applying it; the initial-state Jacobian arrives as a dense
matrix and becomes a dense operator. This is the one place they are
converted.
"""

from typing import Generic, Optional, TypeVar, Union

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    CurvatureFn,
    InitialStateFn,
    Jacobian,
    ODEResidualDerivatives,
    StatedZero,
    dense_operator,
    matrix_free_initial_state_curvature,
    matrix_free_second_derivative,
)
from pyapprox.ode.state_derivatives import StateStateHVPFn
from pyapprox.pde.parameterizations.derivatives import (
    ParamDerivatives,
    ParamHVPFn,
    ParamJacobianFn,
    ParamJacobianProtocol,
)
from pyapprox.pde.parameterizations.initial_value import (
    InitialParamHVPAt,
    InitialParamJacobianAt,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

F = TypeVar("F", bound=FormatProtocol)


class _ParamJacobianOperator(Generic[Array]):
    """A parameter Jacobian's actions between the parameter and state
    spaces; ``MatrixFree``. ``out`` is ignored."""

    def __init__(
        self,
        jacobian: ParamJacobianProtocol[Array],
        params: SpaceDescriptor,
        states: SpaceDescriptor,
    ) -> None:
        self._jacobian = jacobian
        self._params, self._states = params, states

    def domain(self) -> SpaceDescriptor:
        return self._params

    def range(self) -> SpaceDescriptor:
        return self._states

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    @staticmethod
    def _check(block: Array, size: int) -> None:
        if block.ndim != 2 or block.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(block.shape)}"
            )

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, self._params.size)
        return self._jacobian.apply(x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        self._check(y, self._states.size)
        return self._jacobian.apply_transpose(y)


class _ParamJacobianAt(Generic[Array]):
    """``(state, time) -> dF/dp`` at the current parameters, as an
    operator applying the parameterization's actions."""

    def __init__(
        self,
        fn: ParamJacobianFn[Array],
        parameters: ParameterState[Array],
        params: VectorSpaceProtocol[Array],
        states: VectorSpaceProtocol[Array],
    ) -> None:
        self._fn, self._parameters = fn, parameters
        self._params, self._states = params.descriptor(), states.descriptor()

    def __call__(self, state: Array, time: float) -> _ParamJacobianOperator[Array]:
        return _ParamJacobianOperator(
            self._fn(state, time, self._parameters.current()),
            self._params,
            self._states,
        )


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
    *,
    initial_state: Union[ParamDerivatives[Array], StatedZero],
) -> ODEResidualDerivatives[Array, F]:
    r"""The bundle of a right-hand side with state Jacobian ``jacobian``
    and the given parameter and state derivatives, read at the current
    parameters of ``parameters``.

    Its order is decided here, once, by which fields are present: first
    order with ``param_jacobian``; second order when, besides, the three
    parameter contractions, ``state_state_hvp`` and the initial state's
    curvature are all present; the state Jacobian alone without
    ``param_jacobian``. A Jacobian that cannot see the state (:math:`f`
    affine) gives the stated zero state curvature.

    The initial state's derivatives are not :math:`f`'s, so where they
    come from is the caller's explicit choice: a bundle whose
    ``initial_param_jacobian`` (and, for second order,
    ``initial_param_hvp``) they are read from, or
    :data:`~pyapprox.ode.residual_derivatives.STATED_ZERO` for an initial
    state independent of the parameters (a steady problem has none). A
    bundle without ``initial_param_jacobian`` is refused rather than
    dropping the parameter derivatives.

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
    initial_state : ParamDerivatives or STATED_ZERO
        Where the initial state's derivatives come from (see above).

    Raises
    ------
    ValueError
        If ``param_jacobian`` is present and ``initial_state`` is a bundle
        without ``initial_param_jacobian``.
    """
    param_jacobian = param_derivatives.param_jacobian
    if param_jacobian is None:
        return ODEResidualDerivatives.state_only(jacobian)
    param = _ParamJacobianAt(param_jacobian, parameters, params, states)
    initial: Union[InitialStateFn[Array], StatedZero] = STATED_ZERO
    curvature: Union[CurvatureFn[Array], StatedZero, None] = STATED_ZERO
    if not isinstance(initial_state, StatedZero):
        initial_param_jacobian = initial_state.initial_param_jacobian
        if initial_param_jacobian is None:
            raise ValueError(
                "the parameterization has a parameter Jacobian but no "
                "initial_param_jacobian: supply it, or state the initial state "
                "independent of the parameters (initial_state=STATED_ZERO)"
            )
        initial = dense_operator(
            InitialParamJacobianAt(initial_param_jacobian, parameters),
            params,
            states,
            bkd,
        )
        initial_param_hvp = initial_state.initial_param_hvp
        curvature = (
            None
            if initial_param_hvp is None
            else matrix_free_initial_state_curvature(
                InitialParamHVPAt(initial_param_hvp, parameters), params, bkd
            )
        )
    param_param_hvp = param_derivatives.param_param_hvp
    state_param_hvp = param_derivatives.state_param_hvp
    param_state_hvp = param_derivatives.param_state_hvp
    if (
        curvature is None
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
        initial_state_curvature=curvature,
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
