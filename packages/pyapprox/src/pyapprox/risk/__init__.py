"""
Sample-based risk measures and statistics.

This module provides sample statistics for computing expectations and
risk measures over samples. These are used for:
- Prediction OED objectives
- Risk-based optimization
- Conservative surrogate fitting
- Stochastic dominance checks
"""

from .avar import SampleAverageSmoothedAVaR
from .base import SampleStatistic
from .entropic_risk import SampleAverageEntropicRisk
from .exact_avar import ExactAVaR
from .mean import SampleAverageMean
from .mean_plus_stdev import SampleAverageMeanPlusStdev
from .ssd import DisutilitySSD, UtilitySSD
from .var import ValueAtRisk
from .variance import SampleAverageStdev, SampleAverageVariance

__all__ = [
    "SampleStatistic",
    "SampleAverageMean",
    "SampleAverageVariance",
    "SampleAverageStdev",
    "SampleAverageEntropicRisk",
    "SampleAverageSmoothedAVaR",
    "SampleAverageMeanPlusStdev",
    "ExactAVaR",
    "ValueAtRisk",
    "UtilitySSD",
    "DisutilitySSD",
]
