"""Local refinement criteria for hierarchical sparse grids.

This module provides refinement criteria for locally-adaptive sparse grids
that prioritize individual basis functions based on their hierarchical surplus.
"""

from typing import TYPE_CHECKING, Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.local.adaptive import (
        LocallyAdaptiveCombinationSparseGrid,
    )


class LocalHierarchicalRefinementCriteria(Generic[Array]):
    """Refinement criteria based on hierarchical surplus.

    The hierarchical surplus is the difference between the function value
    at a basis function's node and the interpolant value computed without
    that basis function. Larger surpluses indicate regions where refinement
    would improve accuracy.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.local import (
    ...     LocalHierarchicalRefinementCriteria
    ... )
    >>> bkd = NumpyBkd()
    >>> criteria = LocalHierarchicalRefinementCriteria(bkd)
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(
        self,
        basis_index: Array,
        basis_value: Array,
        grid: "LocallyAdaptiveCombinationSparseGrid[Array]",
    ) -> Tuple[float, float]:
        """Compute hierarchical surplus refinement priority.

        Parameters
        ----------
        basis_index : Array
            Local basis index. Shape: (nvars,)
        basis_value : Array
            Function value at basis node. Shape: (nqoi,)
        grid : LocallyAdaptiveCombinationSparseGrid
            The locally adaptive sparse grid.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where:
            - priority: Higher value means refine sooner
            - error: Absolute hierarchical surplus (max over QoIs)
        """
        # Get the sample location for this basis function
        sample = grid._get_basis_sample(basis_index)

        # Evaluate current interpolant at this location
        interpolant_value = grid._evaluate_selected_only(sample)

        # Hierarchical surplus is the difference
        surplus = self._bkd.abs(basis_value - interpolant_value)

        # Error is max surplus over all QoIs
        error: float = self._bkd.max(surplus).item()

        # Higher error = higher priority
        return error, error

    def __repr__(self) -> str:
        return "LocalHierarchicalRefinementCriteria()"
