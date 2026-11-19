r"""Integrands of time integrals, for :class:`~pyapprox.ode.time_integral.
TimeIntegral`.

:class:`WeightedSquaredNorm` is :math:`q = y^\top W y`, whose integral is
the squared :math:`W`-norm of the trajectory over time.
"""

from typing import Generic

from pyapprox.implicitfunction.functional_terms import (
    BackendScalarGradient,
    LinearizationProtocol,
)
from pyapprox.ode.time_integral import IntegrandDerivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.symmetric_weight import SymmetricWeightProtocol


class WeightedSquaredNorm(Generic[Array]):
    r""":math:`q(y) = y^\top W y` for a symmetric positive-semidefinite
    weight :math:`W`, of a backend state column; one entry.

    It reads neither the time nor the parameters. Its state derivative is
    :math:`2 W y` and its weighted state second derivative
    :math:`2\omega W`. The weight is used only through its action, so it
    may be stored in any form (a subdomain-weighted mass, sparse) and is
    never densified here.

    Parameters
    ----------
    weight : SymmetricWeightProtocol
        :math:`W`.
    bkd : Backend
        The backend.

    Raises
    ------
    TypeError
        If ``weight`` does not satisfy ``SymmetricWeightProtocol``.
    """

    def __init__(self, weight: SymmetricWeightProtocol[Array], bkd: Backend[Array]):
        if not isinstance(weight, SymmetricWeightProtocol):
            raise TypeError(
                "weight must satisfy SymmetricWeightProtocol, got "
                f"{type(weight).__name__}; wrap a matrix in DenseMatrixWeight "
                "or ScipySparseWeight"
            )
        self._weight, self._bkd = weight, bkd
        self._derivatives: IntegrandDerivatives[Array, Array] = (
            IntegrandDerivatives.param_free_second_order(
                self._state_derivative, self._state_state
            )
        )

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array, time: float) -> Array:
        """:math:`y^\\top W y`. Shape: (1, 1)."""
        return self._bkd.dot(state.T, self._weight.apply(state))

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        return self._derivatives

    def _state_derivative(
        self, state: Array, time: float
    ) -> LinearizationProtocol[Array, Array]:
        return BackendScalarGradient(2.0 * self._weight.apply(state), self._bkd)

    def _state_state(
        self, state: Array, time: float, weights: Array, x: Array
    ) -> Array:
        return 2.0 * self._weight.apply(x) * weights

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nstates={self._weight.nstates()})"
