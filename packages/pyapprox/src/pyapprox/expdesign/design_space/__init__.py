"""Feasible sets of relaxed design weights, and maps onto the weights.

``BoxBudgetDesignSpace`` satisfies ``DesignSpaceProtocol``: it supplies the
bounds, constraints, budget and starting point a relaxed solver searches
over. ``GroupedDesign`` maps group weights to observation weights, and
``ParameterizedObjective`` turns an objective of the weights into one of
those design variables. ``BinaryDesignSubsetObjective`` scores a subset of
the design variables as the objective at its 0/1 design, and
``ReevaluatingIncremental`` grows such subsets by rescoring.
"""

from .binary import BinaryDesignSubsetObjective, ReevaluatingIncremental
from .box_budget import BoxBudgetDesignSpace
from .grouped import GroupedDesign
from .parameterized import ParameterizedObjective

__all__ = [
    "BinaryDesignSubsetObjective",
    "ReevaluatingIncremental",
    "BoxBudgetDesignSpace",
    "GroupedDesign",
    "ParameterizedObjective",
]
