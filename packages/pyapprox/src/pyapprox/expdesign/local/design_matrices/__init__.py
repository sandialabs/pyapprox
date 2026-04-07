"""
Design matrices for local OED.

This module provides classes for computing design matrices M0 and M1
used in local OED criteria for linear regression models.

Classes
-------
DesignMatricesBase
    Abstract base class for design matrix computation.
LeastSquaresDesignMatrices
    Design matrices for ordinary least squares regression.
QuantileDesignMatrices
    Design matrices for quantile regression.
"""

from .base import DesignMatricesBase
from .least_squares import LeastSquaresDesignMatrices
from .quantile import QuantileDesignMatrices

__all__ = [
    "DesignMatricesBase",
    "LeastSquaresDesignMatrices",
    "QuantileDesignMatrices",
]
