"""
D-vine copula module.

Provides D-vine copula with truncation, precision matrix construction,
and optional integration with Bayesian Gaussian networks.
"""

from pyapprox.probability.copula.vine.dvine import DVineCopula
from pyapprox.probability.copula.vine.helpers import (
    compute_dvine_partial_correlations,
    correlation_from_partial_correlations,
    precision_bandwidth,
)

__all__ = [
    "DVineCopula",
    "precision_bandwidth",
    "compute_dvine_partial_correlations",
    "correlation_from_partial_correlations",
]
