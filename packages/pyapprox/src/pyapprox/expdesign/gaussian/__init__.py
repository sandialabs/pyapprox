"""Gaussian design from the joint moments of targets and observations.

``BlendedObservation`` says how relaxed design weights enter an
observation: it maps weights to the variances of an independent noise
part, so that a zero weight removes a sensor exactly.
"""

from .relaxation import BlendedObservation

__all__ = [
    "BlendedObservation",
]
