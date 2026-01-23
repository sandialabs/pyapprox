"""Sparse grid surrogates using Smolyak combination technique.

This module provides sparse grid interpolation using the Smolyak
combination technique, which combines tensor product interpolants
to achieve efficient high-dimensional approximation.

Key classes:
- CombinationSparseGrid: Base class for sparse grids
- IsotropicCombinationSparseGrid: Pre-computed isotropic sparse grid
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
    LocalIndexGeneratorProtocol,
    LocalRefinementCriteriaProtocol,
)

from .smolyak import (
    compute_smolyak_coefficients,
    is_downward_closed,
    get_subspace_neighbors,
    check_admissibility,
)

from .subspace import TensorProductSubspace

from .combination import CombinationSparseGrid

from .isotropic import IsotropicCombinationSparseGrid

from .adaptive import AdaptiveCombinationSparseGrid

from .multiindex import MultiIndexAdaptiveCombinationSparseGrid

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
    "LocalIndexGeneratorProtocol",
    "LocalRefinementCriteriaProtocol",
    # Smolyak utilities
    "compute_smolyak_coefficients",
    "is_downward_closed",
    "get_subspace_neighbors",
    "check_admissibility",
    # Classes
    "TensorProductSubspace",
    "CombinationSparseGrid",
    "IsotropicCombinationSparseGrid",
    "AdaptiveCombinationSparseGrid",
    "MultiIndexAdaptiveCombinationSparseGrid",
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
