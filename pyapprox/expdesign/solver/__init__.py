"""
OED optimization solvers.

This module provides solvers for optimal experimental design problems,
including continuous relaxation methods and discrete brute-force search.
"""

from .brute_force import BruteForceKLOEDSolver
from .convenience import solve_kl_oed, solve_prediction_oed
from .relaxed import (
    OEDObjectiveWrapper,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    RelaxedOEDSolver,
)

__all__ = [
    "RelaxedOEDSolver",
    "RelaxedKLOEDSolver",
    "RelaxedOEDConfig",
    "OEDObjectiveWrapper",
    "BruteForceKLOEDSolver",
    "solve_kl_oed",
    "solve_prediction_oed",
]
