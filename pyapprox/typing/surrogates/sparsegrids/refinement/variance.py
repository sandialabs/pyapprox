"""Variance-based refinement criteria for adaptive sparse grids.

This module provides refinement criteria based on variance change
for adaptive sparse grid construction.
"""

from typing import TYPE_CHECKING, Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.adaptive import (
        AdaptiveCombinationSparseGrid,
    )


class VarianceRefinementCriteria(Generic[Array]):
    """Refinement criteria based on variance change.

    Estimates the impact of adding a subspace on the interpolant by
    computing the variance of the hierarchical surplus (the difference
    between function values and current interpolant). Subspaces with
    larger variance contributions are prioritized for refinement.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    This is a simplified version that uses the variance of the
    hierarchical surplus as a proxy for the variance contribution.
    For sparse grids with polynomial bases and integration weights,
    the full variance computation would require evaluating moments
    via quadrature.

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

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    @property
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

        The error is computed as the sum of:
        1. Absolute change in mean (L-infinity norm across QoIs)
        2. Square root of absolute change in variance (L-infinity norm)

        Parameters
        ----------
        subspace_index : Array
            Multi-index of the subspace. Shape: (nvars,)
        subspace_values : Array
            Function values at subspace samples. Shape: (nsamples, nqoi)
        grid : AdaptiveCombinationSparseGrid
            The adaptive sparse grid.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where:
            - priority: Higher value means refine sooner (equals error)
            - error: Combined mean and variance change indicator
        """
        from pyapprox.typing.surrogates.sparsegrids.smolyak import _index_to_tuple

        key = _index_to_tuple(subspace_index)
        if key not in grid._subspaces:
            return 0.0, float("inf")

        subspace = grid._subspaces[key]
        samples = subspace.get_samples()

        # Evaluate current interpolant at subspace samples
        current_vals = grid._evaluate_selected_only(samples)

        # Compute hierarchical surplus (difference)
        surplus = subspace_values - current_vals

        # Compute mean and variance of surplus as proxy for contribution
        mean_surplus = self._bkd.mean(surplus, axis=0)
        var_surplus = self._bkd.var(surplus, axis=0)

        # Combined error: |mean change| + sqrt(|variance change|)
        error = float(
            self._bkd.max(self._bkd.abs(mean_surplus))
            + self._bkd.max(self._bkd.sqrt(self._bkd.abs(var_surplus)))
        )

        # Higher error = higher priority
        return error, error

    def __repr__(self) -> str:
        return "VarianceRefinementCriteria()"
