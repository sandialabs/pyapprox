"""Statistics implementations for multifidelity estimation.

This module provides statistic classes that compute quantities of interest
from model samples and their covariance structure.
"""

from pyapprox.typing.stats.statistics.base import AbstractStatistic
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.statistics.variance import MultiOutputVariance
from pyapprox.typing.stats.statistics.mean_variance import MultiOutputMeanAndVariance

__all__ = [
    "AbstractStatistic",
    "MultiOutputMean",
    "MultiOutputVariance",
    "MultiOutputMeanAndVariance",
]
