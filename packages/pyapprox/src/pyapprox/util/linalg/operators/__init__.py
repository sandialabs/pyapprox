"""Library-neutral protocols for vectors, operators and solvers.

Nothing in this package imports a particular linear-algebra or
discretization library; stacks implementing these protocols live
beside it, in :mod:`pyapprox.util.linalg`.
"""

from pyapprox.util.linalg.operators.algebra import (
    Product,
    ScaledSum,
    Transposed,
    as_matrix_free,
    product_properties,
    scaled_sum_properties,
)
from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
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
    "FormatProtocol",
    "Layout",
    "LayoutMismatchError",
    "MatrixFree",
    "OperatorProperties",
    "Product",
    "ScaledSum",
    "Transposed",
    "VectorSpaceProtocol",
    "as_matrix_free",
    "product_properties",
    "scaled_sum_properties",
]
