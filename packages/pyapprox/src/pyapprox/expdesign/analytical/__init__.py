"""
Analytical utilities for OED with conjugate Gaussian priors.

This module provides closed-form formulas for expected deviations
in prediction OED when using linear Gaussian models with conjugate priors.
"""

from .conjugate_gaussian import (
    ConjugateGaussianOEDDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
    ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)
from .lognormal_avar_objective import (
    LogNormalDataMeanQoIAVaRStdDevObjective,
)

__all__ = [
    "ConjugateGaussianOEDPredictionUtilityBase",
    "ConjugateGaussianOEDDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDDataMeanQoIMeanEntropicDev",
    "ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev",
    "ConjugateGaussianOEDDataMeanQoIAVaRStdDev",
    "ConjugateGaussianOEDExpectedKLDivergence",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev",
    "LogNormalDataMeanQoIAVaRStdDevObjective",
]
