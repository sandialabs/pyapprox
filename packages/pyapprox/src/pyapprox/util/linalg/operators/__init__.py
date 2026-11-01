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
from pyapprox.util.linalg.operators.blocks import (
    BlockInnerProductSpace,
    BlockLowerTriangularHandle,
    BlockOperator,
    BlockVector,
    BlockVectorSpace,
)
from pyapprox.util.linalg.operators.caching import (
    CachingSolver,
    KeepWithinSolve,
    KeyedSolverProtocol,
    NoReuse,
    ReuseKey,
    ReusePolicyProtocol,
    UncachedSolver,
    reuse_key,
)
from pyapprox.util.linalg.operators.constraints import (
    RowReplacedOperator,
    RowsZeroedOperator,
    SymmetricEliminatedOperator,
    rows_replaced_properties,
    rows_zeroed_properties,
    symmetric_eliminated_properties,
)
from pyapprox.util.linalg.operators.descriptor import (
    SpaceDescriptor,
    SpaceMismatchError,
)
from pyapprox.util.linalg.operators.formats import FormatProtocol, MatrixFree
from pyapprox.util.linalg.operators.identity import IdentityOperator
from pyapprox.util.linalg.operators.krylov import GMRES, ConjugateGradient
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
from pyapprox.util.linalg.operators.zero import ZeroOperator

__all__ = [
    "ActionOperatorProtocol",
    "BlockInnerProductSpace",
    "BlockLowerTriangularHandle",
    "BlockOperator",
    "BlockVector",
    "BlockVectorSpace",
    "CachingSolver",
    "ConjugateGradient",
    "GMRES",
    "KeepWithinSolve",
    "KeyedSolverProtocol",
    "NoReuse",
    "ReuseKey",
    "ReusePolicyProtocol",
    "UncachedSolver",
    "reuse_key",
    "DiagonalCapableProtocol",
    "DiagonalSolver",
    "FormatProtocol",
    "IdentityOperator",
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
    "SymmetricEliminatedOperator",
    "Transposed",
    "VectorSpaceProtocol",
    "ZeroOperator",
    "InnerProductSpaceProtocol",
    "as_matrix_free",
    "product_properties",
    "rows_replaced_properties",
    "rows_zeroed_properties",
    "scaled_sum_properties",
    "symmetric_eliminated_properties",
]
