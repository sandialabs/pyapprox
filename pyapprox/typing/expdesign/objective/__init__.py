"""
OED objective functions.

This module provides objective functions for optimal experimental design,
including KL-OED (expected information gain) and related objectives.
"""

from .kl_objective import KLOEDObjective
from .parallel_kl_objective import ParallelKLOEDObjective

__all__ = [
    "KLOEDObjective",
    "ParallelKLOEDObjective",
]
