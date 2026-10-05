"""
Protocol definitions for experimental design components.

This module defines the interfaces (protocols) for OED likelihoods,
evidence computation, objectives, quadrature samplers, sample statistics,
deviation measures, prediction objectives, design spaces, observation relaxations, and
subset objectives.
"""

from .design_space import DesignSpaceProtocol
from .deviation import DeviationMeasureProtocol
from .evidence import (
    EvidenceProtocol,
    LogEvidenceProtocol,
)
from .gaussian_criterion import GaussianDesignCriterionProtocol
from .likelihood import (
    OEDInnerLoopLikelihoodProtocol,
    OEDOuterLoopLikelihoodProtocol,
)
from .objective import (
    KLOEDObjectiveProtocol,
    OEDObjectiveProtocol,
)
from .oed import (
    BayesianInferenceProblemProtocol,
    GaussianInferenceProblemProtocol,
    KLOEDProblemProtocol,
    PredictionOEDProblemProtocol,
)
from .prediction import PredictionOEDObjectiveProtocol
from .quadrature import (
    OEDQuadratureSamplerProtocol,
)
from .relaxation import ObservationRelaxationProtocol
from .subset import (
    IncrementalSubsetObjectiveProtocol,
    SubsetObjectiveProtocol,
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
    # Design space protocols
    "DesignSpaceProtocol",
    # Observation relaxation protocols
    "ObservationRelaxationProtocol",
    # Subset objective protocols
    "SubsetObjectiveProtocol",
    "IncrementalSubsetObjectiveProtocol",
    # Gaussian design criterion protocols
    "GaussianDesignCriterionProtocol",
    # Deviation protocols
    "DeviationMeasureProtocol",
    # OED inference/benchmark protocols
    "BayesianInferenceProblemProtocol",
    "GaussianInferenceProblemProtocol",
    "KLOEDProblemProtocol",
    "PredictionOEDProblemProtocol",
]
