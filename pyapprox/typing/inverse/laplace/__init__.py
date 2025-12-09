"""
Laplace approximation module for nonlinear inverse problems.

This module provides Laplace approximation methods for computing
posterior distributions when the observation model is nonlinear.

The Laplace approximation computes a Gaussian approximation to the
posterior by finding the MAP point and using the Hessian of the
negative log-posterior as the precision matrix.
"""

from .hessian_operators import (
    ApplyNegLogLikelihoodHessian,
    PriorConditionedHessianMatVec,
)
from .full_rank import DenseLaplacePosterior
from .low_rank import LowRankLaplacePosterior

__all__ = [
    "ApplyNegLogLikelihoodHessian",
    "PriorConditionedHessianMatVec",
    "DenseLaplacePosterior",
    "LowRankLaplacePosterior",
]
