"""
Protocol definitions for experimental design components.

This module defines the interfaces (protocols) for OED likelihoods,
evidence computation, objectives, and quadrature samplers.
"""

from .likelihood import (
    OEDOuterLoopLikelihoodProtocol,
    OEDInnerLoopLikelihoodProtocol,
)
from .evidence import (
    EvidenceProtocol,
    LogEvidenceProtocol,
)
from .objective import (
    OEDObjectiveProtocol,
    KLOEDObjectiveProtocol,
)
from .quadrature import (
    QuadratureSamplerProtocol,
    OEDQuadratureSamplerProtocol,
)

__all__ = [
    # Likelihood protocols
    "OEDOuterLoopLikelihoodProtocol",
    "OEDInnerLoopLikelihoodProtocol",
    # Evidence protocols
    "EvidenceProtocol",
    "LogEvidenceProtocol",
    # Objective protocols
    "OEDObjectiveProtocol",
    "KLOEDObjectiveProtocol",
    # Quadrature protocols
    "QuadratureSamplerProtocol",
    "OEDQuadratureSamplerProtocol",
]
