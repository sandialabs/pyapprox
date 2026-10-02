"""Feasible sets of relaxed design weights.

Each class satisfies ``DesignSpaceProtocol``: it supplies the bounds,
constraints, budget and starting point a relaxed solver searches over.
"""

from .box_budget import BoxBudgetDesignSpace

__all__ = [
    "BoxBudgetDesignSpace",
]
