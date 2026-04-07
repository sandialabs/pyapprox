"""
Minimax optimization utilities.

This module provides tools for solving minimax optimization problems:
    min_x max_i f_i(x)

The approach introduces a slack variable t and reformulates as:
    min_{x,t} t
    s.t. t >= f_i(x) for all i

This allows using standard constrained optimization solvers.
"""

from .constraint import MinimaxConstraint
from .objective import MinimaxObjective
from .optimizer import MinimaxOptimizer
from .protocols import MultiQoIObjectiveProtocol, SlackBasedObjectiveProtocol

__all__ = [
    # Protocols
    "MultiQoIObjectiveProtocol",
    "SlackBasedObjectiveProtocol",
    # Objective and constraint
    "MinimaxObjective",
    "MinimaxConstraint",
    # Optimizer
    "MinimaxOptimizer",
]
