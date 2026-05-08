"""
Analytical risk measures for probability distributions.

This module provides closed-form risk measures for specific distributions,
useful for testing, validation, and benchmark comparisons.

Currently supported distributions:
- Gaussian (normal)
- LogNormal

For sample-based risk measures, see ``pyapprox.risk``.
"""

from .gaussian import GaussianAnalyticalRiskMeasures
from .lognormal import LogNormalAnalyticalRiskMeasures

__all__ = [
    "GaussianAnalyticalRiskMeasures",
    "LogNormalAnalyticalRiskMeasures",
]
