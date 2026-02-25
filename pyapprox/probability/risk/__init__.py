"""
Risk measures for probability distributions.

This module provides:
- Analytical (closed-form) risk measures for specific distributions
- Sample-based risk measures computed from empirical distributions

Analytical risk measures are useful for:
- Testing and validating sample-based implementations
- Computing exact values when closed-form formulas exist
- Benchmark comparisons

Sample-based risk measures are useful for:
- Computing risk of empirical distributions
- Conservative surrogate fitting (adjusting constant term)
- Risk-based optimization objectives

Currently supported distributions for analytical measures:
- Gaussian (normal)
- LogNormal

Sample-based risk measures:
- SafetyMarginRiskMeasure: mean + strength * std
- ValueAtRisk: empirical quantile
- AverageValueAtRisk: CVaR / expected shortfall
- EntropicRisk: exponential utility risk measure
- UtilitySSD: utility form of second-order stochastic dominance
- DisutilitySSD: disutility form of second-order stochastic dominance
"""

from .gaussian import GaussianAnalyticalRiskMeasures
from .lognormal import LogNormalAnalyticalRiskMeasures
from .measures import (
    RiskMeasureProtocol,
    RiskMeasureBase,
    SafetyMarginRiskMeasure,
    ValueAtRisk,
    AverageValueAtRisk,
    EntropicRisk,
    UtilitySSD,
    DisutilitySSD,
)

__all__ = [
    "GaussianAnalyticalRiskMeasures",
    "LogNormalAnalyticalRiskMeasures",
    "RiskMeasureProtocol",
    "RiskMeasureBase",
    "SafetyMarginRiskMeasure",
    "ValueAtRisk",
    "AverageValueAtRisk",
    "EntropicRisk",
    "UtilitySSD",
    "DisutilitySSD",
]
