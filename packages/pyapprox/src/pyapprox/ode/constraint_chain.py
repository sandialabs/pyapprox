r"""The chain rule through a state constraint, linearized at one point.

A constrained state is :math:`y = \Pi_t(x;\, p)`, :math:`x` the value the
constraint is applied to (an initial state :math:`\hat y_0`, a step's
readout :math:`E u_n`). Every consumer differentiating through
:math:`\Pi` needs the same three operations at that point, so they are
defined once here:

- forward, :math:`dy = \Pi'\, dx + \Pi_p\, v`;
- backward, :math:`\bar x = \Pi'^\top \bar y` and the parameter term
  :math:`\Pi_p^\top \bar y`;
- second order along :math:`(dx, v)` with weight :math:`\bar y`, the
  state part :math:`(\bar y^\top \Pi)_{xx}\, dx + (\bar y^\top \Pi)_{xp}\,
  v` and the parameter part :math:`(\bar y^\top \Pi)_{px}\, dx +
  (\bar y^\top \Pi)_{pp}\, v`.

A term the constraint states zero is skipped, never applied.
"""

from typing import Generic, Optional, Tuple, TypeVar

from pyapprox.implicitfunction.functional_terms import accumulate
from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.util.linalg.operators import VectorSpaceProtocol

V = TypeVar("V")


class LinearizedConstraint(Generic[V]):
    r""":math:`\Pi_t` linearized at the value it was applied to.

    Parameters
    ----------
    derivatives : StateConstraintDerivatives
        The constraint's bundle.
    point : V
        :math:`x`, the value the constraint was applied to.
    time : float
        :math:`t`.
    owner : str
        The constraint's name, for refusals.

    Raises
    ------
    TypeError
        If the bundle lacks ``tangent`` or ``param``.
    """

    def __init__(
        self,
        derivatives: StateConstraintDerivatives[V],
        point: V,
        time: float,
        owner: str,
    ) -> None:
        if derivatives.tangent is None or derivatives.param is None:
            raise TypeError(
                f"{owner}'s derivatives() lacks tangent or param, so it has no "
                "parameter gradients"
            )
        self._derivatives, self._point, self._time = derivatives, point, time
        self._owner = owner
        self._tangent: DerivativeOperator[V] = derivatives.tangent(point, time)
        self._param: DerivativeOperator[V] = derivatives.param(point, time)

    def param_is_zero(self) -> bool:
        r"""Whether the constraint states :math:`\Pi_p` zero."""
        return self._param.properties().zero

    def tangent(self, dx: V, direction: V, space: VectorSpaceProtocol[V]) -> V:
        r""":math:`\Pi'\, dx + \Pi_p\, v`, a new vector."""
        dy = self._tangent.apply(dx)
        if not self._param.properties().zero:
            dy = space.axpby(1.0, self._param.apply(direction), 1.0, dy, out=dy)
        return dy

    def pull(self, cotangent: V) -> Tuple[V, Optional[V]]:
        r""":math:`\Pi'^\top \bar y`, a new vector, and the parameter term
        :math:`\Pi_p^\top \bar y`, None when the constraint states it
        zero."""
        pulled = self._tangent.apply_transpose(cotangent)
        if self._param.properties().zero:
            return pulled, None
        return pulled, self._param.apply_transpose(cotangent)

    def second(
        self,
        cotangent: V,
        dx: V,
        direction: V,
        space: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> Tuple[Optional[V], Optional[V]]:
        r"""The second-order terms along :math:`(dx, v)` weighted by
        ``cotangent`` :math:`\bar y`: the state part
        :math:`(\bar y^\top \Pi)_{xx}\, dx + (\bar y^\top \Pi)_{xp}\, v`
        and the parameter part :math:`(\bar y^\top \Pi)_{px}\, dx +
        (\bar y^\top \Pi)_{pp}\, v`, each None when every term in it is a
        stated zero.

        Raises
        ------
        TypeError
            If the bundle has no second derivatives.
        """
        derivs = self._derivatives
        if (
            derivs.state_state is None
            or derivs.state_param is None
            or derivs.param_param is None
        ):
            raise TypeError(
                f"{self._owner}'s derivatives() has no second derivatives, so it "
                "has no Hessian-vector products"
            )
        point, time = self._point, self._time
        state_state = derivs.state_state(point, time, cotangent)
        state_param = derivs.state_param(point, time, cotangent)
        param_param = derivs.param_param(point, time, cotangent)
        to_state: Optional[V] = None
        for op, x in ((state_state, dx), (state_param, direction)):
            if not op.properties().zero:
                to_state = accumulate(to_state, op.apply(x), space)
        to_param: Optional[V] = None
        if not state_param.properties().zero:
            to_param = accumulate(to_param, state_param.apply_transpose(dx), params)
        if not param_param.properties().zero:
            to_param = accumulate(to_param, param_param.apply(direction), params)
        return to_state, to_param

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._owner}, time={self._time})"
