"""
Univariate marginal distribution implementations.

This module provides implementations of common univariate probability
distributions for use in uncertainty quantification.

Key Classes
-----------
- GaussianMarginal: Gaussian (normal) distribution
- ContinuousScipyRandomVariable1D: Continuous distributions from SciPy
- DiscreteScipyRandomVariable1D: Discrete distributions from SciPy

Examples
--------
>>> from pyapprox.typing.variables.univariate import GaussianMarginal
>>> from pyapprox.typing.util.backends.numpy import NumpyBkd
>>> marginal = GaussianMarginal(mean=0, std=1, bkd=NumpyBkd())
"""

from .gaussian import GaussianMarginal
from .scipy_continuous import ContinuousScipyRandomVariable1D
from .scipy_discrete import DiscreteScipyRandomVariable1D

__all__ = [
    "GaussianMarginal",
    "ContinuousScipyRandomVariable1D",
    "DiscreteScipyRandomVariable1D",
]
