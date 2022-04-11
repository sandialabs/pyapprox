"""The :mod:`pyapprox.bayes` module implements a number of popular tools for
Bayesian Inference.
"""

from pyapprox.bayes.gaussian_network import GaussianNetwork
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models
)


__all__ = ["GaussianNetwork",
           "laplace_posterior_approximation_for_linear_models"]
