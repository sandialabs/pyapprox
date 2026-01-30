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
    DirectSolverResult,
    SparseResult,
    BayesianConjugateFitter,
    BayesianConjugateResult,
)

# Export losses
from pyapprox.typing.surrogates.affine.expansions.losses import (
    BasisExpansionMSELoss,
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
    "DirectSolverResult",
    "SparseResult",
    "BayesianConjugateFitter",
    "BayesianConjugateResult",
    # Losses
    "BasisExpansionMSELoss",
    # Statistics module
    "pce_statistics",
]
