"""
OED objective functions.

This module provides objective functions for optimal experimental design,
including KL-OED (expected information gain) and prediction OED objectives.
"""

from .kl_objective import KLOEDObjective
from .parallel_kl_objective import ParallelKLOEDObjective
from .prediction_objective import PredictionOEDObjective
from .d_optimal_linear import DOptimalLinearModelObjective

__all__ = [
    "KLOEDObjective",
    "ParallelKLOEDObjective",
    "PredictionOEDObjective",
    "DOptimalLinearModelObjective",
]
