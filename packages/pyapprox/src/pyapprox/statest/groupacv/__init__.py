"""Group ACV and MLBLUE estimator implementations.

This module provides Group Approximate Control Variate (GroupACV) and
Multi-Level Best Linear Unbiased Estimator (MLBLUE) implementations.
"""

from pyapprox.statest.groupacv.allocation import (
    GroupACVAllocationOptimizer,
    default_groupacv_optimizer,
)
from pyapprox.statest.groupacv.base import (
    BaseGroupACVEstimator,
    FittedGroupACVEstimator,
)
from pyapprox.statest.groupacv.brute_force import (
    BruteForceSubsetFitter,
    BruteForceSubsetResult,
)
from pyapprox.statest.groupacv.mlblue import (
    MLBLUEEstimator,
)
from pyapprox.statest.groupacv.mlblue_optimizer import (
    MLBLUESPDAllocationOptimizer,
)
from pyapprox.statest.groupacv.optimization import (
    GroupACVCostConstraint,
    GroupACVLogDetObjective,
    GroupACVObjective,
    GroupACVTraceObjective,
    MLBLUEObjective,
)
from pyapprox.statest.groupacv.result import (
    GroupACVAllocationResult,
)
from pyapprox.statest.groupacv.search import (
    GroupACVSearch,
    GroupACVSearchResult,
)
from pyapprox.statest.groupacv.utils import (
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _grouped_acv_sigma,
    _grouped_acv_sigma_block,
    _nest_subsets,
    get_model_subsets,
    get_model_subsets_limited,
)
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
)
from pyapprox.statest.groupacv.variants import (
    GroupACVEstimatorIS,
    GroupACVEstimatorNested,
)

__all__ = [
    # Utility functions
    "get_model_subsets",
    "get_model_subsets_limited",
    "_get_allocation_matrix_is",
    "_get_allocation_matrix_nested",
    "_nest_subsets",
    "_grouped_acv_sigma_block",
    "_grouped_acv_sigma",
    # Optimization classes
    "GroupACVObjective",
    "GroupACVTraceObjective",
    "GroupACVLogDetObjective",
    "MLBLUEObjective",
    "GroupACVCostConstraint",
    # Allocation classes
    "AllocationProblemConfig",
    "GroupACVAllocationResult",
    "GroupACVAllocationOptimizer",
    "default_groupacv_optimizer",
    "MLBLUESPDAllocationOptimizer",
    "BruteForceSubsetFitter",
    "BruteForceSubsetResult",
    # Estimator classes
    "BaseGroupACVEstimator",
    "FittedGroupACVEstimator",
    "GroupACVEstimatorIS",
    "GroupACVEstimatorNested",
    "MLBLUEEstimator",
    # Search classes
    "GroupACVSearch",
    "GroupACVSearchResult",
]
