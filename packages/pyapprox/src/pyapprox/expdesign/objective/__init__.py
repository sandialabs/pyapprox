"""
OED objective functions.

This module provides objective functions for optimal experimental design,
including KL-OED (expected information gain) and prediction OED objectives.
"""

from .d_optimal_linear import DOptimalLinearModelObjective
from .factory import (
    create_kl_oed_objective,
    create_kl_oed_objective_from_data,
)
from .kl_objective import KLOEDObjective
from .parallel_kl_objective import ParallelKLOEDObjective
from .prediction_factory import (
    create_deviation_measure,
    create_prediction_oed_objective,
    create_risk_measure,
)
from .prediction_objective import PredictionOEDObjective

__all__ = [
    "KLOEDObjective",
    "ParallelKLOEDObjective",
    "PredictionOEDObjective",
    "DOptimalLinearModelObjective",
    "create_kl_oed_objective",
    "create_kl_oed_objective_from_data",
    "create_deviation_measure",
    "create_risk_measure",
    "create_prediction_oed_objective",
]
