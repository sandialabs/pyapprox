"""Variance-based refinement criteria for adaptive sparse grids.

This module provides refinement criteria based on mean and variance change
for adaptive sparse grid construction.
"""

from typing import TYPE_CHECKING, Generic, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.sparsegrids.refinement.protocols import (
    SparseGridCostFunctionProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.cost import UnitCostFunction

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.adaptive import (
        AdaptiveCombinationSparseGrid,
    )


class VarianceRefinementCriteria(Generic[Array]):
    """Refinement criteria based on mean and variance change.

    Computes the impact of adding a subspace by computing the change
    in mean and variance that would result from including the candidate.
    Priority = error / cost.

    Error = max(|new_mean - current_mean|) + max(sqrt(|new_var - current_var|))

    This matches the legacy implementation in
    pyapprox.surrogates.sparsegrids.combination.VarianceRefinementCriteria,
    which uses _adjust_smolyak_coefficients_with_candidate_index to compute
    what the moments would be if the candidate were selected.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    cost_function : SparseGridCostFunctionProtocol[Array], optional
        Cost function for cost-aware refinement. Default: UnitCostFunction.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.refinement import (
    ...     VarianceRefinementCriteria
    ... )
    >>> bkd = NumpyBkd()
    >>> criteria = VarianceRefinementCriteria(bkd)
    >>> # Use with AdaptiveCombinationSparseGrid
    >>> # grid = AdaptiveCombinationSparseGrid(..., refinement_priority=criteria)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        cost_function: Optional[SparseGridCostFunctionProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._cost_function: SparseGridCostFunctionProtocol[Array] = (
            cost_function if cost_function is not None else UnitCostFunction(bkd)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(
        self,
        subspace_index: Array,
        subspace_values: Array,
        grid: "AdaptiveCombinationSparseGrid[Array]",
    ) -> Tuple[float, float]:
        """Compute variance-based refinement priority for a subspace.

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
            - priority: Higher value means refine sooner (error / cost)
            - error: Combined mean and variance change indicator
        """
        from pyapprox.typing.surrogates.sparsegrids.smolyak import _index_to_tuple

        key = _index_to_tuple(subspace_index)
        if key not in grid._subspaces:
            return 0.0, float("inf")

        # Get current moments using selected subspaces only
        current_mean = grid.mean()
        current_variance = grid.variance()

        # Compute what moments would be if candidate were added
        new_smolyak_coefs = grid._adjust_smolyak_coefficients_with_candidate_index(
            subspace_index
        )
        new_mean = grid._compute_moment("integrate", new_smolyak_coefs)
        new_variance = grid._compute_moment("variance", new_smolyak_coefs)

        # Error = |mean change| + sqrt(|variance change|)
        # Take max over all QoIs for multi-output functions
        mean_change = self._bkd.max(self._bkd.abs(new_mean - current_mean))
        var_change = self._bkd.max(
            self._bkd.sqrt(self._bkd.abs(new_variance - current_variance))
        )
        error = float(mean_change.item() + var_change.item())

        # Priority = error / cost (following legacy pattern)
        subspace = grid._subspaces[key]
        nsamples = subspace.get_samples().shape[1]
        cost = nsamples * self._cost_function(subspace_index)
        priority = error / max(cost, 1e-14)

        return priority, error

    def __repr__(self) -> str:
        return f"VarianceRefinementCriteria(cost_function={self._cost_function!r})"
