"""
Base class for local OED solvers.

Provides common infrastructure for optimizing design weights subject to
simplex constraints (sum = 1, weights >= 0).
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.expdesign.local.protocols.criterion import (
    LocalOEDCriterionProtocol,
)


class LocalOEDSolverBase(Generic[Array]):
    """
    Base class for local OED solvers.

    Provides common simplex constraint setup and default initialization
    for design weight optimization.

    Parameters
    ----------
    criterion : LocalOEDCriterionProtocol[Array]
        Criterion to optimize (e.g., D-optimal, A-optimal).
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    Design weights must satisfy:
    - sum(weights) = 1 (probability measure)
    - weights >= 0 (non-negative)
    - weights <= 1 (bounded)

    Subclasses must implement `construct()`.
    """

    def __init__(
        self,
        criterion: LocalOEDCriterionProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        self._criterion = criterion
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of design variables (candidate design points)."""
        return self._criterion.nvars()

    def _create_simplex_constraint(self) -> PyApproxLinearConstraint[Array]:
        """
        Create linear constraint for simplex: sum(weights) = 1.

        Returns
        -------
        PyApproxLinearConstraint[Array]
            Equality constraint A @ w = 1 where A is all ones.
        """
        nvars = self.nvars()
        A = self._bkd.ones((1, nvars))
        lb = self._bkd.ones((1,))
        ub = self._bkd.ones((1,))
        return PyApproxLinearConstraint(A, lb, ub, self._bkd)

    def _create_bounds(self) -> Array:
        """
        Create bounds for design weights: 0 <= w <= 1.

        Returns
        -------
        Array
            Bounds array. Shape: (nvars, 2)
        """
        nvars = self.nvars()
        lower = self._bkd.zeros((nvars, 1))
        upper = self._bkd.ones((nvars, 1))
        return self._bkd.hstack([lower, upper])

    def _default_init_weights(self) -> Array:
        """
        Create default initial weights (uniform distribution).

        Returns
        -------
        Array
            Uniform weights. Shape: (nvars, 1)
        """
        nvars = self.nvars()
        return self._bkd.full((nvars, 1), 1.0 / nvars)

    def construct(self, init_weights: Optional[Array] = None) -> Array:
        """
        Find optimal design weights.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nvars, 1)
            If None, uses uniform weights.

        Returns
        -------
        Array
            Optimal design weights. Shape: (nvars, 1)
        """
        raise NotImplementedError("Subclasses must implement construct()")
