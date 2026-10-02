"""Solver-neutral boundary-condition building blocks.

Shared leaf of the pde layering: imports nothing from the solver
packages (collocation, galerkin); both consume it. Besides ``sparse_utils``
it imports only ``constitutive``, for the time-awareness declarations a
``BoundarySignal`` shares with coefficient suppliers.
"""

from pyapprox.pde.boundary.classification import BCDofClassification
from pyapprox.pde.boundary.constraint_set import DirichletConstraintSet
from pyapprox.pde.boundary.natural_operator import NaturalBCOperator
from pyapprox.pde.boundary.protocols import (
    ConstraintSetProtocol,
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.pde.boundary.signal import BoundarySignal, DofSignal

__all__ = [
    "BCDofClassification",
    "BoundarySignal",
    "DofSignal",
    "ConstraintSetProtocol",
    "DirichletConstraintSet",
    "EssentialBCProtocol",
    "NaturalBCOperator",
    "WeakFormBCProtocol",
]
