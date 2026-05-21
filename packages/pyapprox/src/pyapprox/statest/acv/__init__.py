"""Approximate Control Variate (ACV) estimator implementations."""

from pyapprox.statest.acv.allocation import (
    ACVAllocator,
    Allocator,
    AnalyticalAllocator,
    default_allocator_factory,
)
from pyapprox.statest.acv.base import ACVEstimator, FittedACVEstimator
from pyapprox.statest.acv.optimization import (
    ACVLogDeterminantObjective,
    ACVObjective,
    ACVPartitionConstraint,
)
from pyapprox.statest.acv.result import ACVAllocationResult
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
    "ACVObjective",
    "ACVLogDeterminantObjective",
    "ACVPartitionConstraint",
    "ACVEstimator",
    "FittedACVEstimator",
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
