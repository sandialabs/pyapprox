"""Feasible sets of relaxed design weights, and maps onto the weights.

``BoxBudgetDesignSpace`` satisfies ``DesignSpaceProtocol``: it supplies the
bounds, constraints, budget and starting point a relaxed solver searches
over. ``GroupedDesign`` maps group weights to observation weights, and
``ParameterizedObjective`` turns an objective of the weights into one of
those design variables.
"""

from .box_budget import BoxBudgetDesignSpace
from .grouped import GroupedDesign
from .parameterized import ParameterizedObjective

__all__ = [
    "BoxBudgetDesignSpace",
    "GroupedDesign",
    "ParameterizedObjective",
]
