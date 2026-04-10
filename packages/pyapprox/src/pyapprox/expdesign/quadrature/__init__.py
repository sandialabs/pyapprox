"""
Quadrature samplers for OED expectation computation.

This module provides samplers for generating quadrature points and weights
used in computing expectations over prior and data distributions.
Supports Monte Carlo, quasi-Monte Carlo (Halton/Sobol), and Gaussian quadrature.
"""

from pyapprox.util.sampling.halton import HaltonSampler
from pyapprox.util.sampling.sobol import SobolSampler

from .gaussian import GaussianQuadratureSampler
from .monte_carlo import MonteCarloSampler
from .oed import OEDQuadratureSampler
from .strategies import (
    GaussStrategy,
    HaltonStrategy,
    MCStrategy,
    SamplerStrategy,
    SobolStrategy,
    get_sampler,
    list_samplers,
    register_sampler,
)

__all__ = [
    "MonteCarloSampler",
    "HaltonSampler",
    "SobolSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
    # Sampler strategies
    "SamplerStrategy",
    "GaussStrategy",
    "MCStrategy",
    "HaltonStrategy",
    "SobolStrategy",
    "register_sampler",
    "get_sampler",
    "list_samplers",
]
