"""
Multivariate Gaussian distributions.

This package provides multivariate Gaussian distributions with various
covariance representations.

Classes
-------
GaussianLogPDFCore
    Shared computation for Gaussian log-PDF.
DenseCholeskyMultivariateGaussian
    Multivariate Gaussian with dense Cholesky covariance.
DiagonalMultivariateGaussian
    Multivariate Gaussian with diagonal covariance.
OperatorBasedMultivariateGaussian
    Multivariate Gaussian with operator-based covariance.
"""

from .core import GaussianLogPDFCore
from .dense import DenseCholeskyMultivariateGaussian
from .diagonal import DiagonalMultivariateGaussian
from .operator import OperatorBasedMultivariateGaussian

__all__ = [
    "GaussianLogPDFCore",
    "DenseCholeskyMultivariateGaussian",
    "DiagonalMultivariateGaussian",
    "OperatorBasedMultivariateGaussian",
]
