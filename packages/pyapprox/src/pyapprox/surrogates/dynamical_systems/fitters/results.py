"""Fit result containers for dynamical systems learning.

All attributes accessed via methods per CLAUDE.md conventions.
"""

from typing import Generic

from pyapprox.surrogates.dynamical_systems.vector_fields import (
    BasisExpansionVectorField,
)
from pyapprox.util.backends.protocols import Array, Backend


class DirectSolverResult(Generic[Array]):
    """Result from direct linear solver (LeastSquares, Ridge).

    Parameters
    ----------
    surrogate : BasisExpansionVectorField[Array]
        Fitted vector field with coefficients set.
    params : Array
        Fitted coefficients. Shape: (nterms, nqoi)
    """

    def __init__(
        self,
        surrogate: BasisExpansionVectorField[Array],
        params: Array,
    ):
        self._surrogate = surrogate
        self._params = params

    def surrogate(self) -> BasisExpansionVectorField[Array]:
        return self._surrogate

    def params(self) -> Array:
        return self._params

    def bkd(self) -> Backend[Array]:
        return self._surrogate.bkd()

    def __call__(self, states: Array) -> Array:
        return self._surrogate(states)

    def __repr__(self) -> str:
        return f"DirectSolverResult(params_shape={self._params.shape})"
