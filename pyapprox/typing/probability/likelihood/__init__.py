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
"""

from .gaussian import GaussianLogLikelihood, DiagonalGaussianLogLikelihood

__all__ = [
    "GaussianLogLikelihood",
    "DiagonalGaussianLogLikelihood",
]
