"""
Quadrature samplers for OED expectation computation.

This module provides samplers for generating quadrature points and weights
used in computing expectations over prior and data distributions.
Supports Monte Carlo, quasi-Monte Carlo (Halton/Sobol), and Gaussian quadrature.
"""

from .sampler import (
    QuadratureSampler,
    MonteCarloSampler,
    HaltonSampler,
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)

__all__ = [
    "QuadratureSampler",
    "MonteCarloSampler",
    "HaltonSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
]
