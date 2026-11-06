r"""A right-hand side split into partitions: :math:`f = \sum_k f_k`.

Its bundle sums the partitions' state and parameter Jacobians (first
order: what the split-system checks read); the partitions share one
parameter version, given to it.
"""

from typing import Callable, Generic, Sequence

from pyapprox.ode.mass_form_system import RightHandSideProtocol
from pyapprox.ode.residual_derivatives import (
    DerivativeOperator,
    Jacobian,
    ODEResidualDerivatives,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    ScaledSum,
    VectorSpaceProtocol,
)

_Rhs = RightHandSideProtocol[Array, DenseFormat[Array]]


class _SummedJacobian(Generic[Array]):
    def __init__(
        self, parts: Sequence[_Rhs[Array]], states: VectorSpaceProtocol[Array]
    ) -> None:
        self._parts, self._states = parts, states

    def __call__(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        terms = []
        for part in self._parts:
            jacobian = part.derivatives().jacobian
            if jacobian is None:
                raise TypeError("every partition needs a Jacobian")
            terms.append((1.0, jacobian(state, time, point)))
        return LinearizedOperator(ScaledSum(terms, self._states), point)


class _SummedParam(Generic[Array]):
    def __init__(
        self,
        parts: Sequence[_Rhs[Array]],
        states: VectorSpaceProtocol[Array],
        params: VectorSpaceProtocol[Array],
    ) -> None:
        self._parts, self._states, self._params = parts, states, params

    def __call__(self, state: Array, time: float) -> DerivativeOperator[Array]:
        terms = []
        for part in self._parts:
            param = part.derivatives().param
            if param is None:
                raise TypeError("every partition needs a parameter Jacobian")
            terms.append((1.0, param(state, time)))
        return ScaledSum(terms, self._states, self._params)


class SummedRightHandSide(Generic[Array]):
    """The sum of right-hand sides on one state space, with first-order
    derivatives.

    Parameters
    ----------
    parts : sequence of RightHandSideProtocol
        The partitions, each with a state and parameter Jacobian; the
        first's initial-state derivative and parameter space are the
        sum's.
    states : VectorSpaceProtocol
        The state space.
    version : callable
        The partitions' shared parameter version.
    """

    def __init__(
        self,
        parts: Sequence[_Rhs[Array]],
        states: VectorSpaceProtocol[Array],
        version: Callable[[], int],
    ) -> None:
        self._parts, self._version = tuple(parts), version
        first = self._parts[0].derivatives()
        if first.param_space is None or first.initial_state is None:
            raise TypeError("the first partition needs parameter derivatives")
        self._derivatives: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.first_order(
                states,
                first.param_space,
                jacobian=Jacobian.general(_SummedJacobian(self._parts, states)),
                param=_SummedParam(self._parts, states, first.param_space),
                initial_state=first.initial_state,
            )
        )

    def __call__(self, state: Array, time: float) -> Array:
        total = self._parts[0](state, time)
        for part in self._parts[1:]:
            total = total + part(state, time)
        return total

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives

    def parameters_version(self) -> int:
        """The partitions' shared version."""
        return self._version()
