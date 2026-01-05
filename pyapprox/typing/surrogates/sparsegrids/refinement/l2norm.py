"""L2 norm refinement criteria for adaptive sparse grids.

This module provides refinement criteria based on L2 norm error
for adaptive sparse grid construction.
"""

from typing import TYPE_CHECKING, Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.adaptive import (
        AdaptiveCombinationSparseGrid,
    )


class L2NormRefinementCriteria(Generic[Array]):
    """Refinement criteria based on L2 norm interpolation error.

    Computes the error as the L2 norm of the difference between function
    values at subspace samples and the current interpolant evaluated at
    those samples. Higher errors result in higher refinement priority.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.refinement import (
    ...     L2NormRefinementCriteria
    ... )
    >>> bkd = NumpyBkd()
    >>> criteria = L2NormRefinementCriteria(bkd)
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
        """Compute L2 norm refinement priority for a subspace.

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
            - error: L2 norm interpolation error normalized by sample count
        """
        from pyapprox.typing.surrogates.sparsegrids.smolyak import _index_to_tuple

        key = _index_to_tuple(subspace_index)
        if key not in grid._subspaces:
            return 0.0, float("inf")

        subspace = grid._subspaces[key]
        samples = subspace.get_samples()

        # Evaluate current interpolant at subspace samples
        current_vals = grid._evaluate_selected_only(samples)

        # Compute L2 norm error normalized by number of samples
        error = float(
            self._bkd.norm(subspace_values - current_vals)
            / max(1, subspace_values.shape[0])
        )

        # Higher error = higher priority
        return error, error

    def __repr__(self) -> str:
        return "L2NormRefinementCriteria()"
