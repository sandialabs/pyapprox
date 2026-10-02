"""Inference from the joint moments of targets and noisy observations.

``JointGaussian`` treats targets and observations as one Gaussian built
from covariance blocks plus additive noise, and conditions it on data or
restricts it to a subset of the observations. ``observe`` returns a
``LinearGaussianObservation``: one target seen through the relaxed
observation at design weights, computed without square roots.
"""

from .joint_gaussian import JointGaussian
from .observation import LinearGaussianObservation

__all__ = [
    "JointGaussian",
    "LinearGaussianObservation",
]
