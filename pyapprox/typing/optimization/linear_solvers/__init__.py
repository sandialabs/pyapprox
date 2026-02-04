"""Linear system solvers for Ax = b.

This module provides solvers for linear systems:
- Direct solvers (LU factorization)
- Iterative solvers (CG, PCG)
- Preconditioners (Jacobi, Block Jacobi)

Note: For coefficient fitting (finding c such that Phi*c = y),
see pyapprox.typing.optimization.linear instead.
"""

from pyapprox.typing.optimization.linear_solvers.protocols import (
    LinearSolverProtocol,
    IterativeSolverProtocol,
    MatrixFreeSolverProtocol,
    PreconditionerProtocol,
    PreconditionerWithSetupProtocol,
)
from pyapprox.typing.optimization.linear_solvers.direct import (
    DirectSolver,
    direct_solve,
)
from pyapprox.typing.optimization.linear_solvers.iterative import (
    ConjugateGradient,
    cg_solve,
    PreconditionedConjugateGradient,
    pcg_solve,
)
from pyapprox.typing.optimization.linear_solvers.preconditioners import (
    JacobiPreconditioner,
    BlockJacobiPreconditioner,
    jacobi_preconditioner,
    block_jacobi_preconditioner,
)

__all__ = [
    # Protocols
    "LinearSolverProtocol",
    "IterativeSolverProtocol",
    "MatrixFreeSolverProtocol",
    "PreconditionerProtocol",
    "PreconditionerWithSetupProtocol",
    # Direct
    "DirectSolver",
    "direct_solve",
    # Iterative
    "ConjugateGradient",
    "cg_solve",
    "PreconditionedConjugateGradient",
    "pcg_solve",
    # Preconditioners
    "JacobiPreconditioner",
    "BlockJacobiPreconditioner",
    "jacobi_preconditioner",
    "block_jacobi_preconditioner",
]
