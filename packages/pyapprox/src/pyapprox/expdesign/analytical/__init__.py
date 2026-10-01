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
    ConjugateGaussianOEDExpectedInformationGain,
    ConjugateGaussianOEDExpectedPushforwardKLDivergence,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)
from .lognormal_avar_objective import (
    LogNormalDataMeanQoIAVaRStdDevObjective,
)
from .lognormal_avar_saa_objective import (
    LogNormalDataMeanQoIAVaRStdDevSAAObjective,
)
from .lognormal_mg_blocks import (
    LogNormalMGBlocks,
    lognormal_goal_mg_blocks,
)
from .outer_data import (
    MarginalOuterData,
    ReparameterizedOuterData,
)
from .relaxed_linear_gaussian import (
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
    relaxed_lognormal_expected_variance,
)

__all__ = [
    "ConjugateGaussianOEDPredictionUtilityBase",
    "ConjugateGaussianOEDDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDDataMeanQoIMeanEntropicDev",
    "ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev",
    "ConjugateGaussianOEDDataMeanQoIAVaRStdDev",
    "ConjugateGaussianOEDExpectedInformationGain",
    "ConjugateGaussianOEDExpectedPushforwardKLDivergence",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance",
    "ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev",
    "LogNormalDataMeanQoIAVaRStdDevObjective",
    "LogNormalDataMeanQoIAVaRStdDevSAAObjective",
    "LogNormalMGBlocks",
    "lognormal_goal_mg_blocks",
    "relaxed_linear_target_covariance",
    "relaxed_linear_target_eig",
    "relaxed_lognormal_expected_variance",
    "MarginalOuterData",
    "ReparameterizedOuterData",
]
