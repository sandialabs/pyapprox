"""
Analytical risk measures for probability distributions.

This module provides closed-form formulas for risk measures (mean, variance,
AVaR, entropic risk, etc.) for specific probability distributions. These are
useful for:
- Testing and validating sample-based risk measure implementations
- Computing exact values when analytical formulas exist
- Benchmark comparisons

Currently supported distributions:
- Gaussian (normal)

Future additions may include:
- Beta
- LogNormal
- Chi-squared
"""

from .gaussian import GaussianAnalyticalRiskMeasures

__all__ = [
    "GaussianAnalyticalRiskMeasures",
]
