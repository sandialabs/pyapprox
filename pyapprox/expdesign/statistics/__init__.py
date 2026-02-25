"""
Sample average statistics for OED.

This module provides sample statistics for computing expectations and
risk measures over samples. These are used for prediction OED objectives.
"""

from .avar import SampleAverageSmoothedAVaR
from .base import SampleStatistic
from .entropic_risk import SampleAverageEntropicRisk
from .mean import SampleAverageMean
from .mean_plus_stdev import SampleAverageMeanPlusStdev
from .variance import SampleAverageStdev, SampleAverageVariance

__all__ = [
    "SampleStatistic",
    "SampleAverageMean",
    "SampleAverageVariance",
    "SampleAverageStdev",
    "SampleAverageEntropicRisk",
    "SampleAverageSmoothedAVaR",
    "SampleAverageMeanPlusStdev",
]
