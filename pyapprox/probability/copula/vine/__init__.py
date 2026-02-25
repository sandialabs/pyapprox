"""
D-vine copula module.

Provides D-vine copula with truncation, precision matrix construction,
and optional integration with Bayesian Gaussian networks.
"""

from pyapprox.probability.copula.vine.dvine import DVineCopula
from pyapprox.probability.copula.vine.helpers import (
    precision_bandwidth,
    compute_dvine_partial_correlations,
    correlation_from_partial_correlations,
)
from pyapprox.probability.copula.vine.gaussian_network_factory import (
    dvine_from_gaussian_network,
)

__all__ = [
    "DVineCopula",
    "dvine_from_gaussian_network",
    "precision_bandwidth",
    "compute_dvine_partial_correlations",
    "correlation_from_partial_correlations",
]
