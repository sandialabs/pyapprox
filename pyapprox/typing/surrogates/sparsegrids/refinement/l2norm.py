"""L2 norm refinement criteria for adaptive sparse grids.

This module provides refinement criteria based on L2 norm error
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


class L2NormRefinementCriteria(Generic[Array]):
    """Refinement criteria based on L2 norm interpolation error.

    Computes the error as the L2 norm of the hierarchical surplus
    (difference between function values and current interpolant)
    normalized by sample count. Priority = error / cost.

    This matches the legacy implementation in
    pyapprox.surrogates.sparsegrids.combination.L2NormRefinementCriteria.

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
    ...     L2NormRefinementCriteria
    ... )
    >>> bkd = NumpyBkd()
    >>> criteria = L2NormRefinementCriteria(bkd)
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
        """Compute L2 norm refinement priority for a subspace.

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
            - error: L2 norm interpolation error normalized by sample count
        """
        from pyapprox.typing.surrogates.sparsegrids.smolyak import _index_to_tuple

        key = _index_to_tuple(subspace_index)
        if key not in grid._subspaces:
            return 0.0, float("inf")

        subspace = grid._subspaces[key]
        samples = subspace.get_samples()

        # Evaluate current interpolant at subspace samples
        current_vals = grid._evaluate_with_selected_indices(samples)

        # Compute L2 norm error normalized by number of samples
        nsamples = subspace_values.shape[1]
        error = float(
            self._bkd.norm(subspace_values - current_vals) / max(1, nsamples)
        )

        # Priority = error / cost (following legacy pattern)
        cost = nsamples * self._cost_function(subspace_index)
        priority = error / max(cost, 1e-14)

        return priority, error

    def __repr__(self) -> str:
        return f"L2NormRefinementCriteria(cost_function={self._cost_function!r})"
