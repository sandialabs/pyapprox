"""Approximate Control Variate (ACV) estimator implementations."""

from pyapprox.statest.acv.allocation import (
    ACVAllocationResult,
    ACVAllocator,
    Allocator,
    AnalyticalAllocator,
    default_allocator_factory,
)
from pyapprox.statest.acv.base import ACVEstimator
from pyapprox.statest.acv.optimization import (
    ACVLogDeterminantObjective,
    ACVObjective,
    ACVPartitionConstraint,
    _combine_acv_samples,
    _combine_acv_values,
    _get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd,
    _get_allocation_matrix_gmf,
)
from pyapprox.statest.acv.search import (
    ACVSearch,
    SearchResult,
)
from pyapprox.statest.acv.strategies import (
    DefaultRecursionStrategy,
    FixedRecursionStrategy,
    HierarchicalPermutationRecursionStrategy,
    ListRecursionStrategy,
    RecursionIndexStrategy,
    TreeDepthRecursionStrategy,
)
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)

__all__ = [
    "_combine_acv_values",
    "_combine_acv_samples",
    "_get_allocation_matrix_gmf",
    "_get_allocation_matrix_acvis",
    "_get_allocation_matrix_acvrd",
    "ACVObjective",
    "ACVLogDeterminantObjective",
    "ACVPartitionConstraint",
    "ACVEstimator",
    "GMFEstimator",
    "GISEstimator",
    "GRDEstimator",
    "MFMCEstimator",
    "MLMCEstimator",
    # Allocation
    "ACVAllocationResult",
    "Allocator",
    "ACVAllocator",
    "AnalyticalAllocator",
    "default_allocator_factory",
    # Strategies
    "RecursionIndexStrategy",
    "DefaultRecursionStrategy",
    "FixedRecursionStrategy",
    "ListRecursionStrategy",
    "TreeDepthRecursionStrategy",
    "HierarchicalPermutationRecursionStrategy",
    # Search
    "SearchResult",
    "ACVSearch",
]
