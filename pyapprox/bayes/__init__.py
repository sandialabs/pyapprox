"""The :mod:`pyapprox.bayes` module implements a number of popular tools for
Bayesian Inference.
"""

from pyapprox.bayes.gaussian_network import GaussianNetwork
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
)


__all__ = [
    "GaussianNetwork",
    "DenseMatrixLaplacePosteriorApproximation",
    "GaussianPushForward",
]
