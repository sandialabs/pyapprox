"""Sparse grid surrogates using Smolyak combination technique.

This module provides sparse grid interpolation using the Smolyak
combination technique, which combines tensor product interpolants
to achieve efficient high-dimensional approximation.

Key classes:
- CombinationSurrogate: Evaluation-only sparse grid surrogate
- IsotropicSparseGridFitter: Fixed-level sparse grid fitter
- AdaptiveSparseGridFitter: Adaptive sparse grid fitter
- TensorProductSubspace: Individual tensor product in sparse grid

Key functions:
- compute_smolyak_coefficients: Compute combination coefficients
- is_downward_closed: Check index set validity
"""

from .protocols import (
    SubspaceProtocol,
    SubspaceWithDerivativesProtocol,
    SparseGridProtocol,
    SparseGridWithDerivativesProtocol,
    AdaptiveSparseGridProtocol,
)

from .smolyak import (
    compute_smolyak_coefficients,
    is_downward_closed,
    get_subspace_neighbors,
    check_admissibility,
)

from .subspace import TensorProductSubspace

# New fitter/surrogate architecture
from .combination_surrogate import CombinationSurrogate
from .isotropic_fitter import IsotropicSparseGridFitter
from .adaptive_fitter import AdaptiveSparseGridFitter
from .fit_result import (
    IsotropicSparseGridFitResult,
    AdaptiveSparseGridFitResult,
)
from .error_indicators import (
    ErrorIndicatorProtocol,
    L2SurrogateDifferenceIndicator,
    L2NewSamplesIndicator,
    VarianceChangeIndicator,
    CostWeightedIndicator,
)
from .candidate_info import CandidateInfo, ConfigIdx
from .cost_model import CostModelProtocol, ConstantCostModel, ExponentialConfigCostModel
from .subspace_factory import SubspaceFactoryProtocol, TensorProductSubspaceFactory
from .sample_tracker import SampleTracker


from .converters import (
    SparseGridToPCEConverter,
    TensorProductSubspaceToPCEConverter,
)

from .basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PiecewiseFactory,
    PrebuiltBasisFactory,
    get_bounds_from_marginal,
    get_transform_from_marginal,
    create_basis_factories,
    create_bases_from_marginals,
)

__all__ = [
    # Protocols
    "SubspaceProtocol",
    "SubspaceWithDerivativesProtocol",
    "SparseGridProtocol",
    "SparseGridWithDerivativesProtocol",
    "AdaptiveSparseGridProtocol",
    # Smolyak utilities
    "compute_smolyak_coefficients",
    "is_downward_closed",
    "get_subspace_neighbors",
    "check_admissibility",
    # Classes — new architecture
    "TensorProductSubspace",
    "CombinationSurrogate",
    "IsotropicSparseGridFitter",
    "AdaptiveSparseGridFitter",
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
    "SubspaceFactoryProtocol",
    "TensorProductSubspaceFactory",
    "SampleTracker",
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
