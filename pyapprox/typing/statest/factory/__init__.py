"""Factory utilities for statest estimators.

Provides factory functions for finding the best estimator configuration
by searching over estimator types, recursion indices, and model subsets.
"""

from pyapprox.typing.statest.factory.tree_enumeration import (
    ModelTree,
    generate_all_trees,
    get_acv_recursion_indices,
    count_recursion_indices,
)
from pyapprox.typing.statest.factory.registry import (
    register_estimator,
    register_statistic,
    get_registered_estimators,
    create_estimator,
    CandidateResult,
    compute_objective,
)
from pyapprox.typing.statest.factory.best_estimator_factory import (
    BestEstimatorFactory,
)

__all__ = [
    # Tree enumeration
    "ModelTree",
    "generate_all_trees",
    "get_acv_recursion_indices",
    "count_recursion_indices",
    # Registry
    "register_estimator",
    "register_statistic",
    "get_registered_estimators",
    "create_estimator",
    "CandidateResult",
    "compute_objective",
    # Factory
    "BestEstimatorFactory",
]
