"""Likelihood functions for Gaussian Process models."""

from pyapprox.surrogates.gaussianprocess.likelihoods.gaussian import (
    GaussianLikelihood,
)
from pyapprox.surrogates.gaussianprocess.likelihoods.protocol import (
    LikelihoodProtocol,
)

__all__ = [
    "LikelihoodProtocol",
    "GaussianLikelihood",
]
