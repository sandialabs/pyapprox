"""Solvers module for spectral collocation methods."""

from pyapprox.typing.pde.collocation.solvers.direct import (
    DirectSolver,
    direct_solve,
)
from pyapprox.typing.pde.collocation.solvers.iterative import (
    ConjugateGradient,
    cg_solve,
    PreconditionedConjugateGradient,
    pcg_solve,
)
from pyapprox.typing.pde.collocation.solvers.preconditioners import (
    JacobiPreconditioner,
    BlockJacobiPreconditioner,
    jacobi_preconditioner,
    block_jacobi_preconditioner,
)

__all__ = [
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
