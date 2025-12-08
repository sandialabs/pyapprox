"""Multi-index generation for affine surrogates."""

from pyapprox.typing.surrogates.affine.indices.utils import (
    hash_index,
    compute_hyperbolic_indices,
    compute_hyperbolic_level_indices,
    sort_indices_lexiographically,
    argsort_indices_lexiographically,
    indices_pnorm,
)

from pyapprox.typing.surrogates.affine.indices.admissibility import (
    AdmissibilityCriteria,
    MaxLevelCriteria,
    Max1DLevelsCriteria,
    MaxIndicesCriteria,
    CompositeCriteria,
)

from pyapprox.typing.surrogates.affine.indices.growth_rules import (
    IndexGrowthRule,
    LinearGrowthRule,
    DoublePlusOneGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
)

from pyapprox.typing.surrogates.affine.indices.generators import (
    IndexGenerator,
    IterativeIndexGenerator,
    HyperbolicIndexGenerator,
)

from pyapprox.typing.surrogates.affine.indices.priority_queue import (
    PriorityQueue,
)

from pyapprox.typing.surrogates.affine.indices.refinement import (
    CostFunction,
    UnitCostFunction,
    LevelCostFunction,
    ExponentialCostFunction,
    RefinementCriteria,
    LevelRefinementCriteria,
    CostWeightedRefinementCriteria,
)

from pyapprox.typing.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)

from pyapprox.typing.surrogates.affine.indices.adaptive import (
    AdaptiveIndexRefinement,
)

__all__ = [
    # Utilities
    "hash_index",
    "compute_hyperbolic_indices",
    "compute_hyperbolic_level_indices",
    "sort_indices_lexiographically",
    "argsort_indices_lexiographically",
    "indices_pnorm",
    # Admissibility criteria
    "AdmissibilityCriteria",
    "MaxLevelCriteria",
    "Max1DLevelsCriteria",
    "MaxIndicesCriteria",
    "CompositeCriteria",
    # Growth rules
    "IndexGrowthRule",
    "LinearGrowthRule",
    "DoublePlusOneGrowthRule",
    "ConstantGrowthRule",
    "ExponentialGrowthRule",
    # Generators
    "IndexGenerator",
    "IterativeIndexGenerator",
    "HyperbolicIndexGenerator",
    # Priority queue
    "PriorityQueue",
    # Refinement
    "CostFunction",
    "UnitCostFunction",
    "LevelCostFunction",
    "ExponentialCostFunction",
    "RefinementCriteria",
    "LevelRefinementCriteria",
    "CostWeightedRefinementCriteria",
    # Basis generator
    "BasisIndexGenerator",
    # Adaptive refinement
    "AdaptiveIndexRefinement",
]
