"""
OED optimization solvers.

This module provides solvers for optimal experimental design problems,
including continuous relaxation methods, discrete brute-force search,
searches over subsets of design variables, and roundings of relaxed
weights.
"""

from .brute_force import BruteForceKLOEDSolver
from .convenience import solve_kl_oed, solve_prediction_oed
from .relaxed import (
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    RelaxedOEDSolver,
)
from .rounding import TopK
from .subset import (
    ExchangeSubsetSolver,
    ExhaustiveSubsetSolver,
    GreedySubsetSolver,
    SubsetSearchResult,
)

__all__ = [
    "RelaxedOEDSolver",
    "RelaxedKLOEDSolver",
    "RelaxedOEDConfig",
    "BruteForceKLOEDSolver",
    "ExhaustiveSubsetSolver",
    "GreedySubsetSolver",
    "ExchangeSubsetSolver",
    "SubsetSearchResult",
    "TopK",
    "solve_kl_oed",
    "solve_prediction_oed",
]
