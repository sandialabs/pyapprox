r"""The initial value as a parameterization target.

A parameter reaches the initial state the way it reaches a coefficient:
a parameterization writes the datum and reports its derivative. The
datum is :class:`InitialValueData`, which holds :math:`\hat y_0` and no
parameters; :class:`InitialValueParameterization` is the part that writes
:math:`\hat y_0 = u_0(p)` and reports ``initial_param_jacobian`` and
``initial_param_hvp`` for it, so a composite with coefficient parts
declares every way the parameters enter in one place.
:class:`ParameterizedInitialState` is the base initial state a transient
map reads: the datum's value, and the parameterization's initial-state
derivatives as operators at the current parameters.

The parameterization reports the initial-state derivative for all its
parts together (coefficient parts report zero); the value is the
datum's, whichever part writes it.
"""

from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.residual_derivatives import (
    CurvatureFn,
    DerivativeOperator,
    dense_operator,
    matrix_free_initial_state_curvature,
)
from pyapprox.ode.residual_form import ResidualFormProtocol
from pyapprox.ode.trajectory_derivatives import InitialStateDerivatives
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamHVPFn,
    InitialParamJacobianFn,
    ParamDerivatives,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import VectorSpaceProtocol


class InitialValueData(Generic[Array]):
    r""":math:`\hat y_0`, written by a parameterization.

    Plain data: it holds a value and no parameters.

    Parameters
    ----------
    state : Array
        The value until a parameterization writes one. Shape:
        (nstates, 1)

    Raises
    ------
    ValueError
        If ``state`` is not a single column.
    """

    def __init__(self, state: Array) -> None:
        self._check(state, "state")
        self._state = state

    @staticmethod
    def _check(state: Array, name: str) -> None:
        if state.ndim != 2 or state.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape (nstates, 1), got {tuple(state.shape)}"
            )

    def nstates(self) -> int:
        """The number of entries."""
        return int(self._state.shape[0])

    def value(self) -> Array:
        """The value last written. Shape: (nstates, 1)"""
        return self._state

    def set_value(self, state: Array) -> None:
        """Write the value.

        Raises
        ------
        ValueError
            If ``state`` is not of shape ``(nstates, 1)``.
        """
        self._check(state, "state")
        if state.shape[0] != self._state.shape[0]:
            raise ValueError(
                f"state must have {self._state.shape[0]} entries, got {state.shape[0]}"
            )
        self._state = state

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self.nstates()})"


@runtime_checkable
class InitialValueFunctionProtocol(Protocol[Array]):
    r""":math:`u_0(p)` as :class:`InitialValueParameterization` reads it:
    ``(nparams, 1) -> (nstates, 1)``, with its derivatives bundle
    (``jacobian`` for first order, a weighted Hessian-vector product for
    second)."""

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array, /) -> Array: ...

    def derivatives(self) -> Derivatives[Array]: ...


class _ZeroParamJacobian(Generic[Array]):
    """A residual's parameter Jacobian that is zero."""

    def __init__(self, nstates: int, nparams: int, bkd: Backend[Array]) -> None:
        self._nstates, self._nparams, self._bkd = nstates, nparams, bkd

    def apply(self, vecs: Array) -> Array:
        return self._bkd.zeros((self._nstates, vecs.shape[1]))

    def apply_transpose(self, vecs: Array) -> Array:
        return self._bkd.zeros((self._nparams, vecs.shape[1]))

    def matrix(self) -> Array:
        return self._bkd.zeros((self._nstates, self._nparams))


class InitialValueParameterization(Generic[Array]):
    r"""A parameterization part writing :math:`\hat y_0 = u_0(p)`.

    The initial value does not enter the residual, so the part's residual
    derivatives are zero; its initial-state Jacobian and curvature are
    ``function``'s. First order when ``function`` has a ``jacobian``,
    second order when it also has a weighted Hessian-vector product.

    Parameters
    ----------
    target : InitialValueData
        Written by :meth:`apply`.
    function : InitialValueFunctionProtocol
        :math:`u_0`, with ``nqoi`` the target's ``nstates``.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If ``function``'s ``nqoi`` is not the target's ``nstates``.
    """

    def __init__(
        self,
        target: InitialValueData[Array],
        function: InitialValueFunctionProtocol[Array],
    ) -> None:
        if not isinstance(target, InitialValueData):
            raise TypeError(
                f"target must be an InitialValueData, got {type(target).__name__}"
            )
        if not isinstance(function, InitialValueFunctionProtocol):
            raise TypeError(
                "function must satisfy InitialValueFunctionProtocol, got "
                f"{type(function).__name__}"
            )
        if function.nqoi() != target.nstates():
            raise ValueError(
                f"function has {function.nqoi()} outputs, but the target has "
                f"{target.nstates()} entries"
            )
        self._target, self._function = target, function
        self._bkd = function.bkd()
        derivs = function.derivatives()
        self._jacobian = derivs.jacobian
        self._whvp = derivs.resolved_whvp(function.nqoi())
        self._derivs: ParamDerivatives[Array] = (
            ParamDerivatives.none()
            if self._jacobian is None
            else ParamDerivatives.first_order(
                self._param_jacobian, self.initial_param_jacobian
            )
            if self._whvp is None
            else ParamDerivatives.second_order(
                self._param_jacobian,
                self.initial_param_jacobian,
                self._params_contraction,
                self._state_contraction,
                self._params_contraction,
                self.initial_param_hvp,
            )
        )

    def nparams(self) -> int:
        return self._function.nvars()

    def targets(self) -> Tuple[object, ...]:
        return (self._target,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("initial_value",)

    def apply(self, params_1d: Array) -> None:
        """Write :math:`u_0(p)` to the target."""
        self._target.set_value(self._function(params_1d[:, None]))

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return self._derivs

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        r""":math:`\partial u_0/\partial p`. Shape: (nstates, nparams)"""
        if self._jacobian is None:
            raise RuntimeError(
                "initial_param_jacobian is unavailable; check param_derivatives() "
                "before calling"
            )
        return self._jacobian(params_1d[:, None])

    def initial_param_hvp(self, params_1d: Array, weight: Array, vvec: Array) -> Array:
        r""":math:`\sum_i w_i\, (\partial^2 u_{0,i}/\partial p^2)\, v`.
        Shape: (nparams,)"""
        if self._whvp is None:
            raise RuntimeError(
                "initial_param_hvp is unavailable; check param_derivatives() "
                "before calling"
            )
        return self._whvp(params_1d[:, None], vvec[:, None], weight[:, None])[:, 0]

    def _param_jacobian(
        self, state: Array, time: float, params_1d: Array
    ) -> _ZeroParamJacobian[Array]:
        """The residual does not read the initial value."""
        return _ZeroParamJacobian(state.shape[0], self.nparams(), self._bkd)

    def _params_contraction(
        self, state: Array, time: float, params_1d: Array, adj: Array, vec: Array
    ) -> Array:
        """A zero contraction into the parameters."""
        return self._bkd.zeros((self.nparams(),))

    def _state_contraction(
        self, state: Array, time: float, params_1d: Array, adj: Array, vec: Array
    ) -> Array:
        """A zero contraction into the state."""
        return self._bkd.zeros((state.shape[0],))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nparams={self.nparams()})"


class InitialParamJacobianAt(Generic[Array]):
    """``() -> dy_0/dp`` at the current parameters."""

    def __init__(
        self, fn: InitialParamJacobianFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self) -> Array:
        return self._fn(self._parameters.current())


class InitialParamHVPAt(Generic[Array]):
    """``(weight, vec) -> sum_i weight_i (d^2 y_0,i/dp^2) vec`` at the
    current parameters."""

    def __init__(
        self, fn: InitialParamHVPFn[Array], parameters: ParameterState[Array]
    ) -> None:
        self._fn, self._parameters = fn, parameters

    def __call__(self, weight: Array, vec: Array) -> Array:
        return self._fn(self._parameters.current(), weight, vec)


class _ResidualWeight(Generic[Array]):
    """A curvature field taking its weight in the residual's form, called
    with a stack vector."""

    def __init__(
        self, field: CurvatureFn[Array], form: ResidualFormProtocol[Array, Array]
    ) -> None:
        self._field, self._form = field, form

    def __call__(self, weight: Array) -> DerivativeOperator[Array]:
        return self._field(self._form.to_residual(weight))


class ParameterizedInitialState(Generic[Array]):
    r"""The base initial state :math:`\hat y_0(p)` of a parameterized
    problem: the value of ``data``, the derivatives
    (``initial_param_jacobian``, ``initial_param_hvp``) of the
    parameterization of ``parameters``, read at its current parameters.

    Which derivatives exist is read from the parameterization's bundle
    here: none without parameters or without ``initial_param_jacobian``,
    second order with ``initial_param_hvp``.

    Parameters
    ----------
    data : InitialValueData
        :math:`\hat y_0`.
    parameters : ParameterState or None
        The problem's parameters; None for a problem without them.
    space : VectorSpaceProtocol
        The state space.
    param_space : VectorSpaceProtocol or None
        The parameter space; required with ``parameters``.
    form : ResidualFormProtocol
        Converts a stack vector to the residual's form, for the
        curvature's weight.
    bkd : Backend
        The backend of the states.

    Raises
    ------
    TypeError
        If ``data`` is not an ``InitialValueData`` or ``parameters`` not a
        ``ParameterState``.
    ValueError
        If ``parameters`` is given without ``param_space``.
    """

    def __init__(
        self,
        data: InitialValueData[Array],
        parameters: Optional[ParameterState[Array]],
        space: VectorSpaceProtocol[Array],
        param_space: Optional[VectorSpaceProtocol[Array]],
        form: ResidualFormProtocol[Array, Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(data, InitialValueData):
            raise TypeError(
                f"data must be an InitialValueData, got {type(data).__name__}"
            )
        if parameters is not None and not isinstance(parameters, ParameterState):
            raise TypeError(
                f"parameters must be a ParameterState, got {type(parameters).__name__}"
            )
        self._data, self._space = data, space
        self._derivatives: InitialStateDerivatives[Array] = (
            InitialStateDerivatives.none()
        )
        if parameters is None:
            return
        if param_space is None:
            raise ValueError("param_space is required with parameters")
        bundle = parameters.parameterization().param_derivatives()
        if bundle.initial_param_jacobian is None:
            return
        jacobian = dense_operator(
            InitialParamJacobianAt(bundle.initial_param_jacobian, parameters),
            param_space,
            space,
            bkd,
        )
        if bundle.initial_param_hvp is None:
            self._derivatives = InitialStateDerivatives.first_order(jacobian)
            return
        curvature = _ResidualWeight(
            matrix_free_initial_state_curvature(
                InitialParamHVPAt(bundle.initial_param_hvp, parameters),
                param_space,
                bkd,
            ),
            form,
        )
        self._derivatives = InitialStateDerivatives.second_order(jacobian, curvature)

    def value(self) -> Array:
        r""":math:`\hat y_0`, a new vector."""
        return self._space.copy(self._data.value())

    def derivatives(self) -> InitialStateDerivatives[Array]:
        return self._derivatives

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"
