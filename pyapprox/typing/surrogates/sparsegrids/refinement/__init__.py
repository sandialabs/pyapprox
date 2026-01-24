"""Refinement criteria for adaptive sparse grids.

This module provides refinement criteria and cost functions for
adaptive sparse grid construction. Criteria compute priorities for
candidate subspaces to determine refinement order.

Protocols:
    SparseGridCostFunctionProtocol: Protocol for cost functions
    SparseGridRefinementCriteriaProtocol: Protocol for refinement criteria

Cost Functions:
    UnitCostFunction: Unit cost (all subspaces equal)
    LevelCostFunction: Cost proportional to L1 norm of index
    ConfigIndexCostFunction: Cost based on config/fidelity index only

Refinement Criteria:
    L2NormRefinementCriteria: Priority based on L2 norm interpolation error
    VarianceRefinementCriteria: Priority based on mean/variance change

Usage with AdaptiveCombinationSparseGrid:
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.sparsegrids.refinement import (
    ...     L2NormRefinementCriteria,
    ...     LevelCostFunction,
    ... )
    >>> bkd = NumpyBkd()
    >>> cost_fn = LevelCostFunction(bkd)
    >>> criteria = L2NormRefinementCriteria(bkd, cost_function=cost_fn)
    >>> # grid = AdaptiveCombinationSparseGrid(..., refinement_priority=criteria)
"""

from pyapprox.typing.surrogates.sparsegrids.refinement.protocols import (
    SparseGridCostFunctionProtocol,
    SparseGridRefinementCriteriaProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.cost import (
    UnitCostFunction,
    LevelCostFunction,
    ConfigIndexCostFunction,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.l2norm import (
    L2NormRefinementCriteria,
)
from pyapprox.typing.surrogates.sparsegrids.refinement.variance import (
    VarianceRefinementCriteria,
)

__all__ = [
    # Protocols
    "SparseGridCostFunctionProtocol",
    "SparseGridRefinementCriteriaProtocol",
    # Cost functions
    "UnitCostFunction",
    "LevelCostFunction",
    "ConfigIndexCostFunction",
    # Refinement criteria
    "L2NormRefinementCriteria",
    "VarianceRefinementCriteria",
]
