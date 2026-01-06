"""
Sample average statistics for OED.

This module provides sample statistics for computing expectations and
risk measures over samples. These are used for prediction OED objectives.
"""

from .base import SampleStatistic
from .mean import SampleAverageMean
from .variance import SampleAverageVariance, SampleAverageStdev
from .entropic_risk import SampleAverageEntropicRisk
from .avar import SampleAverageSmoothedAVaR

__all__ = [
    "SampleStatistic",
    "SampleAverageMean",
    "SampleAverageVariance",
    "SampleAverageStdev",
    "SampleAverageEntropicRisk",
    "SampleAverageSmoothedAVaR",
]
