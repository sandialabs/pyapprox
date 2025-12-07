"""Iterative solvers for spectral collocation methods."""

from pyapprox.typing.pde.collocation.solvers.iterative.cg import (
    ConjugateGradient,
    cg_solve,
)
from pyapprox.typing.pde.collocation.solvers.iterative.pcg import (
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
