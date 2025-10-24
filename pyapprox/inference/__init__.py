"""The :mod:`pyapprox.inference` module implements a number of popular tools for
Bayesian Inference.
"""

from pyapprox.inference.gaussian_network import GaussianNetwork
from pyapprox.inference.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
)


__all__ = [
    "GaussianNetwork",
    "DenseMatrixLaplacePosteriorApproximation",
    "GaussianPushForward",
]
