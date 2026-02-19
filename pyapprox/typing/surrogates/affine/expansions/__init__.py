"""Basis expansions module.

This module provides basis expansion classes for approximating functions
as linear combinations of basis functions.
"""

from pyapprox.typing.surrogates.affine.expansions.base import BasisExpansion

from pyapprox.typing.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
    create_pce,
    create_pce_from_marginals,
)

from pyapprox.typing.surrogates.affine.expansions import pce_statistics
from pyapprox.typing.surrogates.affine.expansions import pce_arithmetic

from pyapprox.typing.surrogates.affine.expansions.pce_marginalize import (
    PCEDimensionReducer,
)

# Re-export solvers from optimization.linear for backward compatibility
from pyapprox.typing.optimization.linear import (
    LeastSquaresSolver,
    RidgeRegressionSolver,
)

# Export fitters
from pyapprox.typing.surrogates.affine.expansions.fitters import (
    LeastSquaresFitter,
    RidgeFitter,
    BPDNFitter,
    OMPFitter,
    DirectSolverResult,
    SparseResult,
    OMPResult,
    CVSelectionResult,
    BayesianConjugateFitter,
    BayesianConjugateResult,
    PCEDegreeSelectionFitter,
    OMPCVFitter,
    AdaptivePCEFitter,
    AdaptivePCEResult,
)

# Export cross-validation functions
from pyapprox.typing.surrogates.affine.expansions.crossvalidation import (
    leave_one_out_lsq_cross_validation,
    leave_many_out_lsq_cross_validation,
    get_random_k_fold_sample_indices,
    get_cross_validation_rsquared,
)

# Export losses
from pyapprox.typing.surrogates.affine.expansions.losses import (
    BasisExpansionMSELoss,
)

# PCE density estimation
from pyapprox.typing.surrogates.affine.expansions.pce_density import (
    UnivariatePCEDensity,
    composite_gauss_legendre,
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
