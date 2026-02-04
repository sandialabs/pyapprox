"""Solvers for spectral collocation methods.

DEPRECATED: Solvers have moved to pyapprox.typing.optimization.linear_solvers.
This module re-exports for backward compatibility.

Import from the new location instead::

    from pyapprox.typing.optimization.linear_solvers import DirectSolver
"""

# Re-export from new location for backward compatibility
from pyapprox.typing.optimization.linear_solvers import (
    DirectSolver,
    direct_solve,
    ConjugateGradient,
    cg_solve,
    PreconditionedConjugateGradient,
    pcg_solve,
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
