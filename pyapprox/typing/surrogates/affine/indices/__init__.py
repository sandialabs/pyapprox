"""Multi-index generation for affine surrogates."""

from pyapprox.typing.surrogates.affine.indices.utils import (
    hash_index,
    compute_hyperbolic_indices,
    compute_hyperbolic_level_indices,
    sort_indices_lexiographically,
    argsort_indices_lexiographically,
    indices_pnorm,
    compute_downward_closure,
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
    ClenshawCurtisGrowthRule,
    ConstantGrowthRule,
    ExponentialGrowthRule,
    CubicNestedGrowthRule,
    inverse_growth_rule,
)

from pyapprox.typing.surrogates.affine.indices.generators import (
    IndexGenerator,
    IterativeIndexGenerator,
    HyperbolicIndexGenerator,
    IsotropicSparseGridBasisIndexGenerator,
    HyperbolicIndexSequence,
    SparseGridIndexSequence,
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

from pyapprox.typing.surrogates.affine.indices.plot import (
    plot_indices_2d,
    plot_indices_3d,
    format_index_axes,
    plot_index_sets,
)

__all__ = [
    # Utilities
    "hash_index",
    "compute_hyperbolic_indices",
    "compute_hyperbolic_level_indices",
    "sort_indices_lexiographically",
    "argsort_indices_lexiographically",
    "indices_pnorm",
    "compute_downward_closure",
    # Admissibility criteria
    "AdmissibilityCriteria",
    "MaxLevelCriteria",
    "Max1DLevelsCriteria",
    "MaxIndicesCriteria",
    "CompositeCriteria",
    # Growth rules
    "IndexGrowthRule",
    "LinearGrowthRule",
    "ClenshawCurtisGrowthRule",
    "ConstantGrowthRule",
    "ExponentialGrowthRule",
    "CubicNestedGrowthRule",
    "inverse_growth_rule",
    # Generators
    "IndexGenerator",
    "IterativeIndexGenerator",
    "HyperbolicIndexGenerator",
    "IsotropicSparseGridBasisIndexGenerator",
    # Index sequences
    "HyperbolicIndexSequence",
    "SparseGridIndexSequence",
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
    # Plotting
    "plot_indices_2d",
    "plot_indices_3d",
    "format_index_axes",
    "plot_index_sets",
]
