"""
OED objective functions.

This module provides objective functions for optimal experimental design,
including KL-OED (expected information gain) and related objectives.
"""

from .kl_objective import KLOEDObjective

__all__ = [
    "KLOEDObjective",
]
