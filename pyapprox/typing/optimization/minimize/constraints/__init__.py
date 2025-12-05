"""
Constraint protocols and implementations for optimization.

This module provides protocols and implementations for constraints used in
optimization problems, including linear and nonlinear constraints.

Key Protocols
-------------
- LinearConstraintProtocol: Protocol for linear constraints
- NonlinearConstraintProtocol: Protocol for nonlinear constraints
- NonlinearConstraintProtocolWithJacobian: With Jacobian support
- NonlinearConstraintProtocolWithJacobianAndWHVP: With Jacobian and weighted HVP

Key Classes
-----------
- PyApproxLinearConstraint: Linear constraint implementation

Examples
--------
>>> from pyapprox.typing.optimization.minimize.constraints import (
...     PyApproxLinearConstraint
... )
>>> constraint = PyApproxLinearConstraint(A, lb, ub, bkd)
"""

from .protocols import (
    LinearConstraintProtocol,
    NonlinearConstraintProtocol,
    NonlinearConstraintProtocolWithJacobian,
    NonlinearConstraintProtocolWithJacobianAndWHVP,
)
from .linear import PyApproxLinearConstraint
from .validation import validate_linear_constraint

__all__ = [
    "LinearConstraintProtocol",
    "NonlinearConstraintProtocol",
    "NonlinearConstraintProtocolWithJacobian",
    "NonlinearConstraintProtocolWithJacobianAndWHVP",
    "PyApproxLinearConstraint",
    "validate_linear_constraint",
]
