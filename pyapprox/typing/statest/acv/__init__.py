"""Approximate Control Variate (ACV) estimator implementations."""

from pyapprox.typing.statest.acv.optimization import (
    _combine_acv_values,
    _combine_acv_samples,
    _get_allocation_matrix_gmf,
    _get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd,
    ACVObjective,
    ACVLogDeterminantObjective,
    ACVPartitionConstraint,
)
from pyapprox.typing.statest.acv.base import ACVEstimator
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)
from pyapprox.typing.statest.acv.allocation import (
    AllocationResult,
    Allocator,
    ACVAllocator,
    AnalyticalAllocator,
    default_allocator_factory,
)
from pyapprox.typing.statest.acv.strategies import (
    RecursionIndexStrategy,
    DefaultRecursionStrategy,
    FixedRecursionStrategy,
    ListRecursionStrategy,
    TreeDepthRecursionStrategy,
    HierarchicalPermutationRecursionStrategy,
)
from pyapprox.typing.statest.acv.search import (
    SearchResult,
    ACVSearch,
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
    "AllocationResult",
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
