r"""An initial state, its value and its parameter derivatives in one object.

A trajectory starts from :math:`y_0 = \Pi_{t_0}(\hat y_0(p);\, p)`: a base
initial state :math:`\hat y_0(p)` with the initial constraint applied
(essential entries set to their data at :math:`t_0`). The base owns its
value and :math:`D = \partial \hat y_0/\partial p`; the constraint owns
:math:`\Pi` and its derivatives. :class:`ConstrainedInitialState` composes
them, so a transient map reads the start value and the sweeps read
:math:`\partial y_0/\partial p` from one object:

.. math::

    \frac{\partial y_0}{\partial p} = \Pi' D + \Pi_p,

and, for a state column :math:`w`, the curvature
:math:`\sum_i w_i\, \partial^2 y_{0,i}/\partial p^2` applied to :math:`v`,

.. math::

    \hat C(\Pi'^\top w)\, v + D^\top\big((w^\top\Pi)_{xx} D v
    + (w^\top\Pi)_{xp} v\big) + (w^\top\Pi)_{px} D v + (w^\top\Pi)_{pp} v,

with :math:`\hat C` the base's curvature. The chain rule through
:math:`\Pi` is :class:`~pyapprox.ode.constraint_chain.LinearizedConstraint`'s,
linearized at :math:`\hat y_0`.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.functional_terms import accumulate
from pyapprox.ode.constraint_chain import LinearizedConstraint
from pyapprox.ode.integration import DifferentiableStackConstraintProtocol
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.trajectory_derivatives import (
    InitialStateDerivativeProtocol,
    InitialStateDerivatives,
)
from pyapprox.util.linalg.operators import (
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")


@runtime_checkable
class InitialValueProtocol(InitialStateDerivativeProtocol[V], Protocol[V]):
    r"""A base initial state :math:`\hat y_0(p)`: its value and its
    parameter derivatives, owned together."""

    def value(self) -> V:
        r""":math:`\hat y_0` at the parameters currently set, a new
        single-column stack vector."""
        ...


class _ConstrainedJacobian(Generic[V]):
    r""":math:`\Pi' D + \Pi_p`, parameters to state, applied through the
    linearized constraint. ``out`` is ignored."""

    def __init__(
        self,
        at: LinearizedConstraint[V],
        base: DerivativeOperator[V],
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._at, self._base = at, base
        self._space, self._params = space, params
        self._base_zero = base.properties().zero

    def domain(self) -> SpaceDescriptor:
        return self._params.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        if self._base_zero and self._at.param_is_zero():
            return OperatorProperties(zero=True)
        return OperatorProperties(transposable=self._base.properties().transposable)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        dx = (
            self._space.zeros(self._params.ncols(x))
            if self._base_zero
            else self._base.apply(x)
        )
        return self._at.tangent(dx, x, self._space)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        pulled, param_term = self._at.pull(y)
        total: Optional[V] = None
        if not self._base_zero:
            total = accumulate(total, self._base.apply_transpose(pulled), self._params)
        total = accumulate(total, param_term, self._params)
        if total is None:
            return self._params.zeros(self._space.ncols(y))
        return total


class _ConstrainedCurvature(Generic[V]):
    r"""The curvature of :math:`w^\top y_0(p)`, parameters to parameters
    (see the module docstring); symmetric. ``out`` is ignored."""

    def __init__(
        self,
        at: LinearizedConstraint[V],
        base: DerivativeOperator[V],
        base_curvature: DerivativeOperator[V],
        weight: V,
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._at, self._base = at, base
        self._base_curvature, self._weight = base_curvature, weight
        self._space, self._params = space, params

    def domain(self) -> SpaceDescriptor:
        return self._params.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._params.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties(symmetric=True)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        params, base = self._params, self._base
        base_zero = base.properties().zero
        dx = self._space.zeros(params.ncols(x)) if base_zero else base.apply(x)
        to_state, to_param = self._at.second(self._weight, dx, x, self._space, params)
        total: Optional[V] = None
        if not self._base_curvature.properties().zero:
            total = accumulate(total, self._base_curvature.apply(x), params)
        if to_state is not None and not base_zero:
            total = accumulate(total, base.apply_transpose(to_state), params)
        total = accumulate(total, to_param, params)
        if total is None:
            return params.zeros(params.ncols(x))
        return total

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self.apply(y)


class ConstrainedInitialState(Generic[V]):
    r""":math:`y_0 = \Pi_{t_0}(\hat y_0(p);\, p)` with its parameter
    derivatives (see the module docstring).

    Which derivatives exist is fixed here, the weaker of the base's and
    the constraint's: first order when the base has ``param_derivative``
    and the constraint ``tangent`` and ``param``; second order when the
    base also has ``curvature`` and the constraint its three second-order
    fields. Each derivative reads the base and the constraint when called,
    so it is valid for the parameters currently set.

    Parameters
    ----------
    base : InitialValueProtocol
        :math:`\hat y_0(p)`.
    constraint : DifferentiableStackConstraintProtocol
        :math:`\Pi`, the initial constraint.
    time : float
        :math:`t_0`.
    space : VectorSpaceProtocol
        The state space.

    Raises
    ------
    TypeError
        If ``base`` or ``constraint`` does not satisfy its protocol.
    """

    def __init__(
        self,
        base: InitialValueProtocol[V],
        constraint: DifferentiableStackConstraintProtocol[V],
        time: float,
        space: VectorSpaceProtocol[V],
    ) -> None:
        if not isinstance(base, InitialValueProtocol):
            raise TypeError(
                f"base must satisfy InitialValueProtocol, got {type(base).__name__}"
            )
        if not isinstance(constraint, DifferentiableStackConstraintProtocol):
            raise TypeError(
                "constraint must satisfy DifferentiableStackConstraintProtocol, "
                f"got {type(constraint).__name__}"
            )
        self._base, self._constraint = base, constraint
        self._time, self._space = time, space
        based, constrained = base.derivatives(), constraint.derivatives()
        self._derivatives: InitialStateDerivatives[V] = (
            InitialStateDerivatives.none()
            if based.param_derivative is None
            or constrained.tangent is None
            or constrained.param is None
            else InitialStateDerivatives.first_order(self._param_derivative)
            if based.curvature is None or constrained.param_param is None
            else InitialStateDerivatives.second_order(
                self._param_derivative, self._curvature
            )
        )

    def value(self) -> V:
        r""":math:`\Pi_{t_0}(\hat y_0)`, a new vector."""
        return self._constraint.constrain(self._base.value(), self._time)

    def derivatives(self) -> InitialStateDerivatives[V]:
        return self._derivatives

    def _linearized(self) -> LinearizedConstraint[V]:
        return LinearizedConstraint(
            self._constraint.derivatives(),
            self._base.value(),
            self._time,
            type(self._constraint).__name__,
        )

    def _params(self) -> VectorSpaceProtocol[V]:
        params = self._constraint.derivatives().param_space
        if params is None:
            raise RuntimeError(
                "the constraint has no parameter space; check derivatives() "
                "before calling"
            )
        return params

    def _base_param_derivative(self) -> DerivativeOperator[V]:
        field = self._base.derivatives().param_derivative
        if field is None:
            raise RuntimeError(
                "param_derivative is unavailable; check derivatives() before calling"
            )
        return field()

    def _param_derivative(self) -> _ConstrainedJacobian[V]:
        r""":math:`\Pi' D + \Pi_p`."""
        return _ConstrainedJacobian(
            self._linearized(),
            self._base_param_derivative(),
            self._space,
            self._params(),
        )

    def _curvature(self, weight: V) -> _ConstrainedCurvature[V]:
        r"""The curvature of :math:`w^\top y_0(p)` for ``weight``
        :math:`w`."""
        field = self._base.derivatives().curvature
        if field is None:
            raise RuntimeError(
                "curvature is unavailable; check derivatives() before calling"
            )
        at = self._linearized()
        pulled, _ = at.pull(weight)
        return _ConstrainedCurvature(
            at,
            self._base_param_derivative(),
            field(pulled),
            weight,
            self._space,
            self._params(),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._base!r}, {self._constraint!r}, "
            f"time={self._time})"
        )


class _Returns(Generic[V]):
    """A ``param_derivative`` field returning one operator."""

    def __init__(self, op: DerivativeOperator[V]) -> None:
        self._op = op

    def __call__(self) -> DerivativeOperator[V]:
        return self._op


class _ReturnsForAnyWeight(Generic[V]):
    """A ``curvature`` field returning one operator for every weight."""

    def __init__(self, op: DerivativeOperator[V]) -> None:
        self._op = op

    def __call__(self, weight: V) -> DerivativeOperator[V]:
        return self._op


class FixedInitialValue(Generic[V]):
    r"""A base initial state independent of the parameters.

    Its derivatives are stated zeros; without a parameter space (a system
    without parameters) it has none.

    Parameters
    ----------
    state : V
        :math:`\hat y_0`, a single column of ``space``; copied.
    space : VectorSpaceProtocol
        The state space.
    param_space : VectorSpaceProtocol or None
        The parameter space; None for a system without parameters.
    """

    def __init__(
        self,
        state: V,
        space: VectorSpaceProtocol[V],
        param_space: Optional[VectorSpaceProtocol[V]],
    ) -> None:
        self._state, self._space = space.copy(state), space
        self._derivatives: InitialStateDerivatives[V] = (
            InitialStateDerivatives.none()
            if param_space is None
            else InitialStateDerivatives.second_order(
                _Returns(ZeroOperator(param_space, space)),
                _ReturnsForAnyWeight(ZeroOperator(param_space)),
            )
        )

    def value(self) -> V:
        return self._space.copy(self._state)

    def derivatives(self) -> InitialStateDerivatives[V]:
        return self._derivatives

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
