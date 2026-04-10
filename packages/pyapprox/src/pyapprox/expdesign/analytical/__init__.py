"""
Analytical utilities for OED with conjugate Gaussian priors.

This module provides closed-form formulas for expected deviations
in prediction OED when using linear Gaussian models with conjugate priors.
"""

from .conjugate_gaussian import (
    ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
    ConjugateGaussianOEDDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
    ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
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
