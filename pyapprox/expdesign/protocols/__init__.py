"""
Protocol definitions for experimental design components.

This module defines the interfaces (protocols) for OED likelihoods,
evidence computation, objectives, quadrature samplers, sample statistics,
deviation measures, and prediction objectives.
"""

from .deviation import DeviationMeasureProtocol
from .evidence import (
    EvidenceProtocol,
    LogEvidenceProtocol,
)
from .likelihood import (
    OEDInnerLoopLikelihoodProtocol,
    OEDOuterLoopLikelihoodProtocol,
)
from .objective import (
    KLOEDObjectiveProtocol,
    OEDObjectiveProtocol,
)
from .prediction import PredictionOEDObjectiveProtocol
from .quadrature import (
    OEDQuadratureSamplerProtocol,
    QuadratureSamplerProtocol,
)
from .statistics import DifferentiableSampleStatisticProtocol, SampleStatisticProtocol

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
    "DifferentiableSampleStatisticProtocol",
    # Deviation protocols
    "DeviationMeasureProtocol",
]
