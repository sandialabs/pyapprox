"""Reference design criteria for nonlinear models.

``ReferenceAOptimal`` and ``ReferenceExpectedInformationGain`` evaluate the
expected posterior trace and the expected information gain without a
Gaussian closure, by rules the caller chooses over the inputs, the data
and, optionally, the nuisances. They measure what the moment-Gaussian
criteria lose, and their optima come from the same solvers.
"""

from .criteria import ReferenceAOptimal, ReferenceExpectedInformationGain

__all__ = [
    "ReferenceAOptimal",
    "ReferenceExpectedInformationGain",
]
