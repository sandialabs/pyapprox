"""Refinement criteria for adaptive sparse grids.

This module provides refinement criteria for adaptive sparse grid
construction. Criteria compute priorities for candidate subspaces
to determine refinement order.

Classes:
    L2NormRefinementCriteria: Priority based on L2 norm interpolation error
    VarianceRefinementCriteria: Priority based on variance change

Usage with AdaptiveCombinationSparseGrid:
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.refinement import (
    ...     L2NormRefinementCriteria
    ... )
    >>> bkd = NumpyBkd()
    >>> criteria = L2NormRefinementCriteria(bkd)
    >>> # grid = AdaptiveCombinationSparseGrid(..., refinement_priority=criteria)
"""

from pyapprox.typing.surrogates.sparsegrids.refinement.l2norm import (
    L2NormRefinementCriteria,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.variance import (
    VarianceRefinementCriteria,
)

__all__ = [
    "L2NormRefinementCriteria",
    "VarianceRefinementCriteria",
]
