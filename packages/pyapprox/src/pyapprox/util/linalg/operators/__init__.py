"""Library-neutral protocols for vectors, operators and solvers.

Nothing in this package imports a particular linear-algebra or
discretization library; stacks implementing these protocols live
beside it, in :mod:`pyapprox.util.linalg`.
"""

from pyapprox.util.linalg.operators.layout import Layout, LayoutMismatchError
from pyapprox.util.linalg.operators.protocols import VectorSpaceProtocol

__all__ = [
    "Layout",
    "LayoutMismatchError",
    "VectorSpaceProtocol",
]
