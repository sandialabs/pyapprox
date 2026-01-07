"""
Likelihood functions module.

This module provides likelihood functions for Bayesian inference
and optimal experimental design.

Classes
-------
GaussianLogLikelihood
    Gaussian likelihood with noise covariance operator.
DiagonalGaussianLogLikelihood
    Gaussian likelihood with diagonal (independent) noise.
ParallelDiagonalGaussianLogLikelihood
    Parallel version with support for multi-process evaluation.
"""

from .gaussian import GaussianLogLikelihood, DiagonalGaussianLogLikelihood
from .parallel_diagonal_gaussian import ParallelDiagonalGaussianLogLikelihood

__all__ = [
    "GaussianLogLikelihood",
    "DiagonalGaussianLogLikelihood",
    "ParallelDiagonalGaussianLogLikelihood",
]
