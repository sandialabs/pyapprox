"""
Joint probability distributions module.

This module provides classes for multivariate joint distributions
composed of multiple marginal distributions.

Classes
-------
IndependentJoint
    Joint distribution with independent marginals.
"""

from .independent import IndependentJoint

__all__ = [
    "IndependentJoint",
]
