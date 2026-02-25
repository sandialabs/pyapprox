"""
AVaR (Average Value at Risk) optimization utilities.

This module provides tools for solving AVaR optimization problems:
    min_x AVaR_alpha[f_i(x)]

where AVaR_alpha is the average of the worst (1-alpha) fraction of outcomes.

The approach introduces slack variables and reformulates as:
    min_{t, s, x} t + (1/(n*(1-alpha))) * sum(s_i)
    s.t. s_i + t >= f_i(x) for all i
         s_i >= 0

This allows using standard constrained optimization solvers.

Notes
-----
AVaR (also known as CVaR or Expected Shortfall) is a coherent risk measure
that quantifies the expected loss given that the loss exceeds the VaR.

For alpha -> 1, AVaR approaches the maximum (minimax case).
For alpha = 0, AVaR equals the mean.
"""

from .protocols import AVaRSlackObjectiveProtocol
from .objective import AVaRObjective
from .constraint import AVaRConstraint
from .optimizer import AVaROptimizer

__all__ = [
    # Protocols
    "AVaRSlackObjectiveProtocol",
    # Objective and constraint
    "AVaRObjective",
    "AVaRConstraint",
    # Optimizer
    "AVaROptimizer",
]
