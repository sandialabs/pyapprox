"""Inference from the joint moments of targets and noisy observations.

``JointGaussian`` treats targets and observations as one Gaussian built
from covariance blocks plus additive noise, and conditions it on data or
restricts it to a subset of the observations.
"""

from .joint_gaussian import JointGaussian

__all__ = [
    "JointGaussian",
]
