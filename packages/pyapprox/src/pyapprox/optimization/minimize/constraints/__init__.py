"""
Constraint protocols and implementations for optimization.

This module provides protocols and implementations for constraints used in
optimization problems, including linear and nonlinear constraints.

Key Protocols
-------------
- LinearConstraintProtocol: Protocol for linear constraints
- NonlinearConstraintProtocol lives at its canonical location,
  ``pyapprox.interface.functions.protocols.objective``

Key Classes
-----------
- PyApproxLinearConstraint: Linear constraint implementation

Examples
--------
>>> from pyapprox.optimization.minimize.constraints import (
...     PyApproxLinearConstraint
... )
>>> constraint = PyApproxLinearConstraint(A, lb, ub, bkd)
"""

from .linear import PyApproxLinearConstraint
from .protocols import (
    LinearConstraintProtocol,
)
from .validation import validate_linear_constraint

__all__ = [
    "LinearConstraintProtocol",
    "PyApproxLinearConstraint",
    "validate_linear_constraint",
]
