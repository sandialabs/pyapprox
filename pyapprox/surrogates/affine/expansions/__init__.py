"""Basis expansions module.

This module provides basis expansion classes for approximating functions
as linear combinations of basis functions.
"""

# Re-export solvers from optimization.linear for backward compatibility
from pyapprox.optimization.linear import (
    LeastSquaresSolver,
    RidgeRegressionSolver,
)
from pyapprox.surrogates.affine.expansions import (
    pce_arithmetic,
    pce_statistics,
)
from pyapprox.surrogates.affine.expansions.base import BasisExpansion

# Export cross-validation functions
from pyapprox.surrogates.affine.expansions.crossvalidation import (
    get_cross_validation_rsquared,
    get_random_k_fold_sample_indices,
    leave_many_out_lsq_cross_validation,
    leave_one_out_lsq_cross_validation,
)

# Export fitters
from pyapprox.surrogates.affine.expansions.fitters import (
    AdaptivePCEFitter,
    AdaptivePCEResult,
    BayesianConjugateFitter,
    BayesianConjugateResult,
    BPDNFitter,
    CVSelectionResult,
    DirectSolverResult,
    LeastSquaresFitter,
    OMPCVFitter,
    OMPFitter,
    OMPResult,
    PCEDegreeSelectionFitter,
    RidgeFitter,
    SparseResult,
)

# Export losses
from pyapprox.surrogates.affine.expansions.losses import (
    BasisExpansionMSELoss,
)
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
    create_pce,
    create_pce_from_marginals,
)

# PCE density estimation
from pyapprox.surrogates.affine.expansions.pce_density import (
    UnivariatePCEDensity,
    composite_gauss_legendre,
)
from pyapprox.surrogates.affine.expansions.pce_marginalize import (
    PCEDimensionReducer,
)

__all__ = [
    # Base classes
    "BasisExpansion",
    # PCE
    "PolynomialChaosExpansion",
    "create_pce",
    "create_pce_from_marginals",
    # Solvers (re-exported for convenience)
    "LeastSquaresSolver",
    "RidgeRegressionSolver",
    # Fitters
    "LeastSquaresFitter",
    "RidgeFitter",
    "BPDNFitter",
    "OMPFitter",
    "DirectSolverResult",
    "SparseResult",
    "OMPResult",
    "BayesianConjugateFitter",
    "BayesianConjugateResult",
    "CVSelectionResult",
    "PCEDegreeSelectionFitter",
    "OMPCVFitter",
    "AdaptivePCEFitter",
    "AdaptivePCEResult",
    # Cross-validation functions
    "leave_one_out_lsq_cross_validation",
    "leave_many_out_lsq_cross_validation",
    "get_random_k_fold_sample_indices",
    "get_cross_validation_rsquared",
    # Losses
    "BasisExpansionMSELoss",
    # PCE density
    "UnivariatePCEDensity",
    "composite_gauss_legendre",
    # Marginalization
    "PCEDimensionReducer",
    # Statistics module
    "pce_statistics",
    # Arithmetic module
    "pce_arithmetic",
]
