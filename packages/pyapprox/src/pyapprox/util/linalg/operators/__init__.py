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
from pyapprox.util.linalg.operators.caching import CachingSolver, ReuseKey, reuse_key
from pyapprox.util.linalg.operators.constraints import (
    RowReplacedOperator,
    RowsZeroedOperator,
    rows_replaced_properties,
    rows_zeroed_properties,
)
from pyapprox.util.linalg.operators.descriptor import (
    SpaceDescriptor,
    SpaceMismatchError,
)
from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
from pyapprox.util.linalg.operators.linearization import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
)
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    InnerProductSpaceProtocol,
    LinearMapProtocol,
    OperatorProperties,
    RowSetProtocol,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.operators.solvers import (
    DiagonalSolver,
    LinearSolverProtocol,
    SolveFailedError,
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
)

__all__ = [
    "ActionOperatorProtocol",
    "CachingSolver",
    "ReuseKey",
    "reuse_key",
    "DiagonalCapableProtocol",
    "DiagonalSolver",
    "FormatProtocol",
    "JacobianSnapshotProtocol",
    "LinearizationPoint",
    "LinearizedOperator",
    "SpaceDescriptor",
    "SpaceMismatchError",
    "LinearMapProtocol",
    "LinearSolverProtocol",
    "MatrixFree",
    "OperatorProperties",
    "Product",
    "RowReplacedOperator",
    "RowSetProtocol",
    "RowsZeroedOperator",
    "ScaledSum",
    "SolveFailedError",
    "SolveInfo",
    "SolverCapabilities",
    "SolverHandleBase",
    "SolverHandleProtocol",
    "Transposed",
    "VectorSpaceProtocol",
    "InnerProductSpaceProtocol",
    "as_matrix_free",
    "product_properties",
    "rows_replaced_properties",
    "rows_zeroed_properties",
    "scaled_sum_properties",
]
