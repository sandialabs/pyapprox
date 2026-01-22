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

# Re-export solvers from solvers module for backward compatibility
from pyapprox.typing.surrogates.affine.solvers import (
    LeastSquaresSolver,
    RidgeRegressionSolver,
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
    # Statistics module
    "pce_statistics",
]
