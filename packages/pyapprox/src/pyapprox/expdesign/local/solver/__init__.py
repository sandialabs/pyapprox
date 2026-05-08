"""
Local OED solvers.

This module provides solvers for finding optimal experimental designs
by minimizing criteria subject to simplex constraints.

Classes
-------
LocalOEDSolverBase
    Base class with simplex constraint setup.
ScipyLocalOEDSolver
    For scalar criteria (D, A, C, I-optimal).
MinimaxLocalOEDSolver
    For G-optimal designs (minimax prediction variance).
AVaRLocalOEDSolver
    For R-optimal designs (risk-based AVaR).
"""

from .avar_solver import AVaRLocalOEDSolver
from .base import LocalOEDSolverBase
from .minimax_solver import MinimaxLocalOEDSolver
from .scipy_solver import ScipyLocalOEDSolver

__all__ = [
    "LocalOEDSolverBase",
    "ScipyLocalOEDSolver",
    "MinimaxLocalOEDSolver",
    "AVaRLocalOEDSolver",
]
