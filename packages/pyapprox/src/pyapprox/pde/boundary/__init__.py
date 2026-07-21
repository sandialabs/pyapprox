"""Solver-neutral boundary-condition building blocks.

Shared leaf of the pde layering: imports nothing from the solver
packages (collocation, galerkin); both consume it.
"""

from pyapprox.pde.boundary.classification import BCDofClassification

__all__ = ["BCDofClassification"]
