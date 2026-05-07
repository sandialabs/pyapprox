"""Sparse grid surrogates: combination technique and hierarchical.

Combination technique (Smolyak):
- CombinationSurrogate, IsotropicSparseGridFitter
- MultiFidelityAdaptiveSparseGridFitter, SingleFidelityAdaptiveSparseGridFitter

Hierarchical (h-adaptive):
- HierarchicalSurrogate, HierarchicalBasis1D, HierarchicalBasisND
- MultiFidelityHierarchicalFitter, SingleFidelityHierarchicalFitter
"""

from .adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
    SubsetType,
)
from .basis.hierarchical_basis_1d import HierarchicalBasis1D
from .basis.hierarchical_basis_nd import HierarchicalBasisND
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
    ErrorIndicatorProtocol,
    L2GlobalSurplusIndicator,
    L2SurplusIndicator,
    VarianceChangeIndicator,
)
from .fit_result import (
    AdaptiveSparseGridFitResult,
    IsotropicSparseGridFitResult,
)
from .hierarchical.error_indicators import (
    GammaIndicator,
    HierarchicalErrorIndicator,
    L2SurplusPointIndicator,
)
from .hierarchical.hierarchical_fitter import (
    MultiFidelityHierarchicalFitter,
    SingleFidelityHierarchicalFitter,
)
from .hierarchical.hierarchical_surrogate import HierarchicalSurrogate
from .isotropic_fitter import IsotropicSparseGridFitter
from .model_factory import (
    DictModelFactory,
    ModelFactoryProtocol,
    TimedModelFactory,
)
from .plot import plot_sparse_grid_points
from .quadrature_rule import ParameterizedIsotropicSparseGridQuadratureRule
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
    "L2GlobalSurplusIndicator",
    "L2SurplusIndicator",
    "VarianceChangeIndicator",
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
    # Quadrature rule
    "ParameterizedIsotropicSparseGridQuadratureRule",
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
    # Hierarchical sparse grids
    "HierarchicalBasis1D",
    "HierarchicalBasisND",
    "HierarchicalSurrogate",
    "MultiFidelityHierarchicalFitter",
    "SingleFidelityHierarchicalFitter",
    "HierarchicalErrorIndicator",
    "GammaIndicator",
    "L2SurplusPointIndicator",
]
