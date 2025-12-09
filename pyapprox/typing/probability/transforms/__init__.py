"""
Probability transforms module.

This module provides transforms for mapping between probability distributions,
including:
- Affine transforms for location-scale families
- Gaussian transforms to/from standard normal
- Rosenblatt and Nataf transforms for correlated variables

Classes
-------
AffineTransform
    Affine (linear) transform: y = (x - loc) / scale.
GaussianTransform
    Transform to/from standard normal using CDF/inverse CDF.
IndependentGaussianTransform
    Transform independent marginals to standard normal.
"""

from .affine import AffineTransform
from .gaussian import GaussianTransform, IndependentGaussianTransform

__all__ = [
    "AffineTransform",
    "GaussianTransform",
    "IndependentGaussianTransform",
]
