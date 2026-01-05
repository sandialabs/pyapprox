"""
Univariate probability distributions.

This package provides univariate distributions that implement the
MarginalProtocol interface.

Classes
-------
BetaMarginal
    Beta distribution with analytical implementation.
GammaMarginal
    Gamma distribution with analytical implementation.
GaussianMarginal
    Gaussian (normal) distribution with analytical implementation.
UniformMarginal
    Uniform distribution with analytical implementation.
ScipyContinuousMarginal
    Wrapper for SciPy continuous distributions.
ScipyDiscreteMarginal
    Wrapper for SciPy discrete distributions.
"""

from .beta import BetaMarginal
from .gamma import GammaMarginal
from .gaussian import GaussianMarginal
from .uniform import UniformMarginal
from .scipy_continuous import ScipyContinuousMarginal
from .scipy_discrete import ScipyDiscreteMarginal

__all__ = [
    "BetaMarginal",
    "GammaMarginal",
    "GaussianMarginal",
    "UniformMarginal",
    "ScipyContinuousMarginal",
    "ScipyDiscreteMarginal",
]
