"""Refinement criteria for adaptive index generation.

This module provides refinement criteria classes that determine which
indices should be refined based on error estimates and priorities.
Used in adaptive sparse grids and PCE.

Criteria:
- LevelRefinementCriteria: Priority by L1 norm of index
- CostWeightedRefinementCriteria: Priority by error/cost ratio
"""

from abc import ABC, abstractmethod
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class CostFunction(ABC, Generic[Array]):
    """Abstract base class for refinement cost functions.

    Cost functions estimate the computational cost of refining a given
    multi-index.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    @property
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    @abstractmethod
    def __call__(self, index: Array) -> float:
        """Compute the cost of refining an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        float
            Estimated cost of refining this index.
        """
        raise NotImplementedError


class UnitCostFunction(CostFunction[Array], Generic[Array]):
    """Unit cost function: all indices have cost 1.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __call__(self, index: Array) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "UnitCostFunction()"


class LevelCostFunction(CostFunction[Array], Generic[Array]):
    """Cost proportional to L1 norm of index.

    Cost = sum(index) + 1

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __call__(self, index: Array) -> float:
        return self._bkd.to_float(self._bkd.sum(index)) + 1.0

    def __repr__(self) -> str:
        return "LevelCostFunction()"


class ExponentialCostFunction(CostFunction[Array], Generic[Array]):
    """Cost exponential in L1 norm.

    Cost = base^(sum(index))

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    base : float
        Base of exponential. Default: 2.0.
    """

    def __init__(self, bkd: Backend[Array], base: float = 2.0):
        super().__init__(bkd)
        self._base = base

    def __call__(self, index: Array) -> float:
        return self._base ** self._bkd.to_float(self._bkd.sum(index))

    def __repr__(self) -> str:
        return f"ExponentialCostFunction(base={self._base})"


class RefinementCriteria(ABC, Generic[Array]):
    """Abstract base class for refinement criteria.

    Refinement criteria compute error estimates and priorities for
    multi-indices, used to decide which indices to refine.
    """

    def __init__(self, bkd: Backend[Array], cost_function: CostFunction[Array]):
        self._bkd = bkd
        self._cost_function = cost_function

    @property
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def cost(self, index: Array) -> float:
        """Compute the cost of refining an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        float
            Estimated cost of refining this index.
        """
        return self._cost_function(index)

    @abstractmethod
    def __call__(self, index: Array) -> Tuple[float, float]:
        """Compute error estimate and priority for an index.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)

        Returns
        -------
        Tuple[float, float]
            (error_estimate, priority).
        """
        raise NotImplementedError


class LevelRefinementCriteria(RefinementCriteria[Array], Generic[Array]):
    """Refinement criteria based on L1 norm of index.

    Priority = max_level - L1_norm(index)

    Higher priority for lower level indices (breadth-first).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_level : int
        Maximum level (for priority computation).
    cost_function : CostFunction[Array], optional
        Cost function. Default: UnitCostFunction.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_level: int,
        cost_function: CostFunction[Array] = None,
    ):
        if cost_function is None:
            cost_function = UnitCostFunction(bkd)
        super().__init__(bkd, cost_function)
        self._max_level = max_level

    def __call__(self, index: Array) -> Tuple[float, float]:
        level = self._bkd.to_float(self._bkd.sum(index))
        # Error estimate: just use level as proxy
        error = level + 1.0
        # Priority: higher for lower levels
        priority = self._max_level - level
        return error, priority

    def __repr__(self) -> str:
        return f"LevelRefinementCriteria(max_level={self._max_level})"


class CostWeightedRefinementCriteria(RefinementCriteria[Array], Generic[Array]):
    """Refinement criteria weighted by cost.

    Priority = error / cost

    Balances error reduction with computational cost.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    cost_function : CostFunction[Array]
        Cost function for computing refinement cost.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        cost_function: CostFunction[Array] = None,
    ):
        if cost_function is None:
            cost_function = UnitCostFunction(bkd)
        super().__init__(bkd, cost_function)
        self._error_estimates: dict = {}

    def set_error_estimate(self, index_id: int, error: float) -> None:
        """Set error estimate for an index.

        Parameters
        ----------
        index_id : int
            Identifier for the index.
        error : float
            Error estimate.
        """
        self._error_estimates[index_id] = error

    def get_error_estimate(self, index_id: int) -> float:
        """Get error estimate for an index.

        Parameters
        ----------
        index_id : int
            Identifier for the index.

        Returns
        -------
        float
            Error estimate, or 0.0 if not set.
        """
        return self._error_estimates.get(index_id, 0.0)

    def __call__(self, index: Array, index_id: int = -1) -> Tuple[float, float]:
        """Compute error estimate and priority.

        Parameters
        ----------
        index : Array
            Multi-index to evaluate. Shape: (nvars,)
        index_id : int, optional
            Index identifier for looking up stored error.

        Returns
        -------
        Tuple[float, float]
            (error_estimate, priority).
        """
        error = self.get_error_estimate(index_id)
        cost = self.cost(index)
        # Priority = error / cost (higher error per unit cost = higher priority)
        priority = error / max(cost, 1e-14)
        return error, priority

    def __repr__(self) -> str:
        return "CostWeightedRefinementCriteria()"
