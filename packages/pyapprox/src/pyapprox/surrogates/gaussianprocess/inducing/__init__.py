"""Inducing point infrastructure for variational Gaussian Processes."""

from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)
from pyapprox.surrogates.gaussianprocess.inducing.variational_distribution import (
    GaussianVariationalDistribution,
)

__all__ = [
    "InducingPoints",
    "GaussianVariationalDistribution",
]
