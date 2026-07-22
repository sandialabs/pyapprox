"""Solver-neutral boundary-condition building blocks.

Shared leaf of the pde layering: imports nothing from the solver
packages (collocation, galerkin); both consume it.
"""

from pyapprox.pde.boundary.classification import BCDofClassification
from pyapprox.pde.boundary.constraint_set import DirichletConstraintSet
from pyapprox.pde.boundary.protocols import (
    ConstraintSetProtocol,
    EssentialBCProtocol,
    WeakFormBCProtocol,
)

__all__ = [
    "BCDofClassification",
    "ConstraintSetProtocol",
    "DirichletConstraintSet",
    "EssentialBCProtocol",
    "WeakFormBCProtocol",
]
