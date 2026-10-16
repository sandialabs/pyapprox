"""Library-neutral protocols for vectors, operators and solvers.

Nothing in this package imports a particular linear-algebra or
discretization library; stacks implementing these protocols live
beside it, in :mod:`pyapprox.util.linalg`.
"""

from pyapprox.util.linalg.operators.formats import MatrixFree
from pyapprox.util.linalg.operators.layout import Layout, LayoutMismatchError
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    OperatorProperties,
    VectorSpaceProtocol,
)

__all__ = [
    "ActionOperatorProtocol",
    "DiagonalCapableProtocol",
    "Layout",
    "LayoutMismatchError",
    "MatrixFree",
    "OperatorProperties",
    "VectorSpaceProtocol",
]
