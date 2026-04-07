"""
Analytical utilities for OED with conjugate Gaussian priors.

This module provides closed-form formulas for expected deviations
in prediction OED when using linear Gaussian models with conjugate priors.
"""

from .conjugate_gaussian import (
    ConjugateGaussianOEDAVaROfExpectedStdDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)

__all__ = [
    "ConjugateGaussianOEDPredictionUtilityBase",
    "ConjugateGaussianOEDExpectedStdDev",
    "ConjugateGaussianOEDExpectedEntropicDev",
    "ConjugateGaussianOEDExpectedAVaRDev",
    "ConjugateGaussianOEDAVaROfExpectedStdDev",
    "ConjugateGaussianOEDExpectedKLDivergence",
    "ConjugateGaussianOEDForLogNormalExpectedStdDev",
    "ConjugateGaussianOEDForLogNormalAVaRStdDev",
]
