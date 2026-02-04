"""Iterative solvers for spectral collocation methods.

DEPRECATED: Moved to pyapprox.typing.optimization.linear_solvers.iterative.
"""

# Re-export from new location for backward compatibility
from pyapprox.typing.optimization.linear_solvers.iterative import (
    ConjugateGradient,
    cg_solve,
    PreconditionedConjugateGradient,
    pcg_solve,
)

__all__ = [
    # Conjugate Gradient
    "ConjugateGradient",
    "cg_solve",
    # Preconditioned CG
    "PreconditionedConjugateGradient",
    "pcg_solve",
]
