"""Protocols for sparse grid refinement criteria.

This module defines protocols for cost functions and refinement criteria
used in adaptive sparse grid construction.
"""

from typing import TYPE_CHECKING, Generic, Protocol, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.adaptive import (
        AdaptiveCombinationSparseGrid,
    )


@runtime_checkable
class SparseGridCostFunctionProtocol(Protocol, Generic[Array]):
    """Protocol for sparse grid cost functions.

    Cost functions estimate the computational cost of evaluating a subspace,
    used to normalize refinement priorities. This enables cost-aware
    adaptive strategies that prioritize inexpensive subspaces.

    Examples
    --------
    >>> class MyCostFunction:
    ...     def __init__(self, bkd):
    ...         self._bkd = bkd
    ...     def bkd(self):
    ...         return self._bkd
    ...     def __call__(self, subspace_index):
    ...         return float(self._bkd.sum(subspace_index)) + 1.0
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(self, subspace_index: Array) -> float:
        """Compute cost per sample for a subspace index.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace. Shape: (nvars,)

        Returns
        -------
        float
            Cost per sample for this subspace (must be positive).
        """
        ...


@runtime_checkable
class SparseGridRefinementCriteriaProtocol(Protocol, Generic[Array]):
    """Protocol for sparse grid refinement criteria.

    Refinement criteria compute error estimates and priorities for
    candidate subspaces, used to decide which subspaces to refine next.
    The priority determines refinement order (higher = refine sooner).

    The __call__ signature takes subspace_index, subspace_values, and grid,
    returning (priority, error) tuple. This matches the interface expected
    by AdaptiveCombinationSparseGrid.

    Examples
    --------
    >>> class MyRefinementCriteria:
    ...     def __init__(self, bkd):
    ...         self._bkd = bkd
    ...     def bkd(self):
    ...         return self._bkd
    ...     def __call__(self, subspace_index, subspace_values, grid):
    ...         error = float(self._bkd.norm(subspace_values))
    ...         return error, error  # priority, error
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(
        self,
        subspace_index: Array,
        subspace_values: Array,
        grid: "AdaptiveCombinationSparseGrid[Array]",
    ) -> Tuple[float, float]:
        """Compute refinement priority for a subspace.

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace. Shape: (nvars,)
        subspace_values : Array
            Function values at subspace samples. Shape: (nqoi, nsamples)
        grid : AdaptiveCombinationSparseGrid
            The adaptive sparse grid.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where:
            - priority: Higher value means refine sooner
            - error: Error estimate for this subspace
        """
        ...
