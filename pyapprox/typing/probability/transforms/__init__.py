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
NatafTransform
    Transform correlated non-Gaussian to independent standard normal.
RosenblattTransform
    General transform using conditional CDFs.
"""

from .affine import AffineTransform
from .gaussian import GaussianTransform, IndependentGaussianTransform
from .nataf import NatafTransform
from .rosenblatt import RosenblattTransform

__all__ = [
    "AffineTransform",
    "GaussianTransform",
    "IndependentGaussianTransform",
    "NatafTransform",
    "RosenblattTransform",
]
