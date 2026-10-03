"""Gaussian design from the joint moments of targets and observations.

``BlendedObservation`` says how relaxed design weights enter an
observation: it maps weights to the variances of an independent noise
part, so that a zero weight removes a sensor exactly. ``AOptimal``,
``DOptimal`` and ``ExpectedInformationGain`` score a target seen through
that observation, with gradients in the weights and variances.
"""

from .criteria import AOptimal, DOptimal, ExpectedInformationGain
from .relaxation import BlendedObservation

__all__ = [
    "AOptimal",
    "BlendedObservation",
    "DOptimal",
    "ExpectedInformationGain",
]
