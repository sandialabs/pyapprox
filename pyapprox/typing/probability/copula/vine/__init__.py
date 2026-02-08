"""
D-vine copula module.

Provides D-vine copula with truncation, precision matrix construction,
and optional integration with Bayesian Gaussian networks.
"""

from pyapprox.typing.probability.copula.vine.dvine import DVineCopula
from pyapprox.typing.probability.copula.vine.helpers import (
    precision_bandwidth,
    compute_dvine_partial_correlations,
    correlation_from_partial_correlations,
)

__all__ = [
    "DVineCopula",
    "precision_bandwidth",
    "compute_dvine_partial_correlations",
    "correlation_from_partial_correlations",
]
