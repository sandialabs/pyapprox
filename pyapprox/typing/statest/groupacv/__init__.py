"""Group ACV and MLBLUE estimator implementations.

This module provides Group Approximate Control Variate (GroupACV) and
Multi-Level Best Linear Unbiased Estimator (MLBLUE) implementations.
"""

from pyapprox.typing.statest.groupacv.utils import (
    get_model_subsets,
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
    _grouped_acv_sigma_block,
    _grouped_acv_sigma,
)

from pyapprox.typing.statest.groupacv.optimization import (
    GroupACVObjective,
    GroupACVTraceObjective,
    GroupACVLogDetObjective,
    MLBLUEObjective,
    GroupACVCostConstraint,
)

from pyapprox.typing.statest.groupacv.base import (
    GroupACVEstimator,
    default_groupacv_optimizer,
)

from pyapprox.typing.statest.groupacv.mlblue import (
    MLBLUEEstimator,
)

__all__ = [
    # Utility functions
    "get_model_subsets",
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
    # Estimator classes
    "GroupACVEstimator",
    "default_groupacv_optimizer",
    "MLBLUEEstimator",
]
