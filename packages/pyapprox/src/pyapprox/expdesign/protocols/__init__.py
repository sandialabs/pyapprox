"""
Protocol definitions for experimental design components.

This module defines the interfaces (protocols) for OED likelihoods,
evidence computation, objectives, quadrature samplers, sample statistics,
deviation measures, and prediction objectives.
"""

from .deviation import DeviationMeasureProtocol
from .oed import (
    BayesianInferenceProblemProtocol,
    GaussianInferenceProblemProtocol,
    KLOEDProblemProtocol,
    PredictionOEDProblemProtocol,
)
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
    "PredictionOEDObjectiveProtocol",
    # Quadrature protocols
    "OEDQuadratureSamplerProtocol",
    # Deviation protocols
    "DeviationMeasureProtocol",
    # OED inference/benchmark protocols
    "BayesianInferenceProblemProtocol",
    "GaussianInferenceProblemProtocol",
    "KLOEDProblemProtocol",
    "PredictionOEDProblemProtocol",
]
