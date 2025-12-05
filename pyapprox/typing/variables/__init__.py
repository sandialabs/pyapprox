"""
Random variable and probability distribution module.

This module provides classes and protocols for working with random variables,
probability distributions, and uncertainty quantification.

Key Classes
-----------
- IndependentRandomVariable: Random variable with independent marginals
- IndependentRandomVariableWithJacobian: With Jacobian support
- IndependentMarginalsVariable: Joint distribution from independent marginals
- GaussianMarginal: Gaussian (normal) marginal distribution
- ContinuousScipyRandomVariable1D: 1D continuous distribution from SciPy
- DiscreteScipyRandomVariable1D: 1D discrete distribution from SciPy

Key Protocols
-------------
- MarginalProtocol: Protocol for marginal distributions
- MarginalWithJacobianProtocol: Marginal with Jacobian support

Examples
--------
>>> from pyapprox.typing.variables import IndependentRandomVariable
>>> from pyapprox.typing.variables.univariate import GaussianMarginal
>>> marginal = GaussianMarginal(0, 1, bkd)
>>> rv = IndependentRandomVariable([marginal])
"""

from .independent import (
    IndependentRandomVariable,
    IndependentRandomVariableWithJacobian,
    MarginalProtocol,
    MarginalWithJacobianProtocol,
)
from .joint import IndependentMarginalsVariable

__all__ = [
    "IndependentRandomVariable",
    "IndependentRandomVariableWithJacobian",
    "MarginalProtocol",
    "MarginalWithJacobianProtocol",
    "IndependentMarginalsVariable",
]
