"""Sparse grid surrogates using Smolyak combination technique.

This module provides sparse grid interpolation using the Smolyak
combination technique, which combines tensor product interpolants
to achieve efficient high-dimensional approximation.

Key classes:
- CombinationSurrogate: Evaluation-only sparse grid surrogate
- IsotropicSparseGridFitter: Fixed-level sparse grid fitter
- MultiFidelityAdaptiveSparseGridFitter: Adaptive MF sparse grid fitter
- SingleFidelityAdaptiveSparseGridFitter: Adaptive SF sparse grid fitter
- TensorProductSubspace: Individual tensor product in sparse grid

Key functions:
- compute_smolyak_coefficients: Compute combination coefficients
- is_downward_closed: Check index set validity
"""

from .adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
    SubsetType,
)
from .basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PiecewiseFactory,
    PrebuiltBasisFactory,
    create_bases_from_marginals,
    create_basis_factories,
    get_bounds_from_marginal,
    get_transform_from_marginal,
)
from .candidate_info import CandidateInfo, ConfigIdx

# New fitter/surrogate architecture
from .combination_surrogate import CombinationSurrogate
from .converters import (
    SparseGridToPCEConverter,
    TensorProductSubspaceToPCEConverter,
)
from .cost_model import (
    ConstantCostModel,
    CostModelProtocol,
    ExponentialConfigCostModel,
    MeasuredCostModel,
)
from .error_indicators import (
    CostWeightedIndicator,
    ErrorIndicatorProtocol,
    L2NewSamplesIndicator,
    L2SurrogateDifferenceIndicator,
    VarianceChangeIndicator,
)
from .fit_result import (
    AdaptiveSparseGridFitResult,
    IsotropicSparseGridFitResult,
)
from .isotropic_fitter import IsotropicSparseGridFitter
from .model_factory import (
    DictModelFactory,
    ModelFactoryProtocol,
    TimedModelFactory,
)
from .plot import plot_sparse_grid_points
from .sample_tracker import SampleTracker
from .smolyak import (
    check_admissibility,
    compute_smolyak_coefficients,
    get_subspace_neighbors,
    is_downward_closed,
)
from .subspace import TensorProductSubspace
from .subspace_factory import (
    SubspaceFactoryProtocol,
    TensorProductSubspaceFactory,
)

__all__ = [
    # Smolyak utilities
    "compute_smolyak_coefficients",
    "is_downward_closed",
    "get_subspace_neighbors",
    "check_admissibility",
    # Classes — new architecture
    "TensorProductSubspace",
    "CombinationSurrogate",
    "IsotropicSparseGridFitter",
    "MultiFidelityAdaptiveSparseGridFitter",
    "SingleFidelityAdaptiveSparseGridFitter",
    "SubsetType",
    "IsotropicSparseGridFitResult",
    "AdaptiveSparseGridFitResult",
    # Error indicators
    "ErrorIndicatorProtocol",
    "L2SurrogateDifferenceIndicator",
    "L2NewSamplesIndicator",
    "VarianceChangeIndicator",
    "CostWeightedIndicator",
    # Data classes and utilities
    "CandidateInfo",
    "ConfigIdx",
    "CostModelProtocol",
    "ConstantCostModel",
    "ExponentialConfigCostModel",
    "MeasuredCostModel",
    # Model factories
    "ModelFactoryProtocol",
    "DictModelFactory",
    "TimedModelFactory",
    "SubspaceFactoryProtocol",
    "TensorProductSubspaceFactory",
    "SampleTracker",
    # Plotting
    "plot_sparse_grid_points",
    # Converters
    "SparseGridToPCEConverter",
    "TensorProductSubspaceToPCEConverter",
    # Basis factories
    "BasisFactoryProtocol",
    "GaussLagrangeFactory",
    "LejaLagrangeFactory",
    "PiecewiseFactory",
    "PrebuiltBasisFactory",
    "get_bounds_from_marginal",
    "get_transform_from_marginal",
    "create_basis_factories",
    "create_bases_from_marginals",
]
