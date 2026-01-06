"""
Protocol definitions for experimental design components.

This module defines the interfaces (protocols) for OED likelihoods,
evidence computation, objectives, quadrature samplers, sample statistics,
deviation measures, and prediction objectives.
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
from .statistics import SampleStatisticProtocol
from .deviation import DeviationMeasureProtocol
from .prediction import PredictionOEDObjectiveProtocol

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
    "PredictionOEDObjectiveProtocol",
    # Quadrature protocols
    "QuadratureSamplerProtocol",
    "OEDQuadratureSamplerProtocol",
    # Statistics protocols
    "SampleStatisticProtocol",
    # Deviation protocols
    "DeviationMeasureProtocol",
]
