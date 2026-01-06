"""
Experimental Design module for pyapprox.typing.

This module provides optimal experimental design (OED) functionality,
with a focus on Bayesian OED using expected information gain (KL divergence).

Submodules
----------
protocols
    Protocol definitions for OED components.
likelihood
    OED-specific likelihood wrappers.
evidence
    Evidence computation for Bayesian OED.
objective
    OED objective functions (KL-OED, etc.).
quadrature
    Quadrature samplers for expectation computation.
solver
    OED optimization solvers.
"""

from .likelihood import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)
from .evidence import Evidence, LogEvidence
from .objective import KLOEDObjective
from .quadrature import (
    QuadratureSampler,
    MonteCarloSampler,
    HaltonSampler,
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)

__all__ = [
    # Likelihood
    "GaussianOEDOuterLoopLikelihood",
    "GaussianOEDInnerLoopLikelihood",
    # Evidence
    "Evidence",
    "LogEvidence",
    # Objective
    "KLOEDObjective",
    # Quadrature samplers
    "QuadratureSampler",
    "MonteCarloSampler",
    "HaltonSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
]
