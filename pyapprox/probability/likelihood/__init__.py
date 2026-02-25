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
MultiExperimentLogLikelihood
    Sum of log-likelihoods across multiple experiments.
ParallelDiagonalGaussianLogLikelihood
    Parallel version with support for multi-process evaluation.
"""

from .gaussian import (
    GaussianLogLikelihood,
    DiagonalGaussianLogLikelihood,
    MultiExperimentLogLikelihood,
)
from .model_based import ModelBasedLogLikelihood
from .parallel_diagonal_gaussian import ParallelDiagonalGaussianLogLikelihood

__all__ = [
    "GaussianLogLikelihood",
    "DiagonalGaussianLogLikelihood",
    "MultiExperimentLogLikelihood",
    "ModelBasedLogLikelihood",
    "ParallelDiagonalGaussianLogLikelihood",
]
