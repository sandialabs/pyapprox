"""Solver-neutral boundary-condition building blocks.

Shared leaf of the pde layering: imports nothing from the solver
packages (collocation, galerkin); both consume it. Within pde it imports
only ``sparse_utils`` and ``constitutive`` (for the time-awareness
declarations a ``BoundarySignal`` shares with coefficient suppliers);
outside pde, ``interface.functions`` so time derivatives can be checked
with ``DerivativeChecker``.
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
from pyapprox.pde.boundary.time_derivative_function import (
    TimeDerivativeFunction,
    time_derivative_functions,
)

__all__ = [
    "BCDofClassification",
    "BoundarySignal",
    "DofSignal",
    "TimeDerivativeFunction",
    "time_derivative_functions",
    "ConstraintSetProtocol",
    "DirichletConstraintSet",
    "EssentialBCProtocol",
    "NaturalBCOperator",
    "WeakFormBCProtocol",
]
