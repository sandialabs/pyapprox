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
from pyapprox.util.linalg.operators.constraints import (
    RowReplacedOperator,
    RowsZeroedOperator,
    rows_replaced_properties,
    rows_zeroed_properties,
)
from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
from pyapprox.util.linalg.operators.layout import Layout, LayoutMismatchError
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    OperatorProperties,
    RowSetProtocol,
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
    "RowReplacedOperator",
    "RowSetProtocol",
    "RowsZeroedOperator",
    "ScaledSum",
    "Transposed",
    "VectorSpaceProtocol",
    "as_matrix_free",
    "product_properties",
    "rows_replaced_properties",
    "rows_zeroed_properties",
    "scaled_sum_properties",
]
