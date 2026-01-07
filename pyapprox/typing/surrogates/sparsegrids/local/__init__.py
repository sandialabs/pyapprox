"""Locally adaptive sparse grid components.

This module provides components for locally-adaptive sparse grids
that refine individual basis functions rather than entire subspaces.

Classes:
    LocalIndexGenerator: Manages hierarchical basis function indices
    LocalHierarchicalRefinementCriteria: Prioritizes basis refinement
    LocallyAdaptiveCombinationSparseGrid: Locally adaptive sparse grid
"""

from pyapprox.typing.surrogates.sparsegrids.local.index_generator import (
    LocalIndexGenerator,
)
from pyapprox.typing.surrogates.sparsegrids.local.refinement import (
    LocalHierarchicalRefinementCriteria,
)
from pyapprox.typing.surrogates.sparsegrids.local.adaptive import (
    LocallyAdaptiveCombinationSparseGrid,
)

__all__ = [
    "LocalIndexGenerator",
    "LocalHierarchicalRefinementCriteria",
    "LocallyAdaptiveCombinationSparseGrid",
]
