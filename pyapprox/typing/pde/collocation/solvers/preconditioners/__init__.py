"""Preconditioners for iterative solvers."""

from pyapprox.typing.pde.collocation.solvers.preconditioners.jacobi import (
    JacobiPreconditioner,
    BlockJacobiPreconditioner,
    jacobi_preconditioner,
    block_jacobi_preconditioner,
)

__all__ = [
    # Jacobi
    "JacobiPreconditioner",
    "BlockJacobiPreconditioner",
    "jacobi_preconditioner",
    "block_jacobi_preconditioner",
]
