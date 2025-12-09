"""
Univariate probability distributions.

This package provides univariate distributions that implement the
MarginalProtocol interface.

Classes
-------
GaussianMarginal
    Gaussian (normal) distribution with analytical implementation.
ScipyContinuousMarginal
    Wrapper for SciPy continuous distributions.
ScipyDiscreteMarginal
    Wrapper for SciPy discrete distributions.
"""

from .gaussian import GaussianMarginal
from .scipy_continuous import ScipyContinuousMarginal
from .scipy_discrete import ScipyDiscreteMarginal

__all__ = [
    "GaussianMarginal",
    "ScipyContinuousMarginal",
    "ScipyDiscreteMarginal",
]
