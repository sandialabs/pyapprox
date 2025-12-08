"""Basis expansions module.

This module provides basis expansion classes for approximating functions
as linear combinations of basis functions.
"""

from pyapprox.typing.surrogates.affine.expansions.base import BasisExpansion

from pyapprox.typing.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
    create_pce,
)

from pyapprox.typing.surrogates.affine.expansions.solvers import (
    LeastSquaresSolver,
    WeightedLeastSquaresSolver,
    RidgeRegressionSolver,
)

from pyapprox.typing.surrogates.affine.expansions import pce_statistics

__all__ = [
    # Base classes
    "BasisExpansion",
    # PCE
    "PolynomialChaosExpansion",
    "create_pce",
    # Solvers
    "LeastSquaresSolver",
    "WeightedLeastSquaresSolver",
    "RidgeRegressionSolver",
    # Statistics module
    "pce_statistics",
]
