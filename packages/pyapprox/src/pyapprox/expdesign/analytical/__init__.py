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
    ConjugateGaussianOEDForLogNormalQoIAVaRDataMeanStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)
from .lognormal_avar_objective import (
    LogNormalQoIAVaRDataMeanStdDevObjective,
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
    "ConjugateGaussianOEDForLogNormalQoIAVaRDataMeanStdDev",
    "LogNormalQoIAVaRDataMeanStdDevObjective",
]
