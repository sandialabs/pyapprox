"""Solver protocols for spectral collocation methods.

DEPRECATED: Protocols have moved to pyapprox.typing.optimization.linear_solvers.
This module re-exports for backward compatibility.

Import from the new location instead::

    from pyapprox.typing.optimization.linear_solvers import LinearSolverProtocol
"""

# Re-export from new location for backward compatibility
from pyapprox.typing.optimization.linear_solvers.protocols import (
    LinearSolverProtocol,
    IterativeSolverProtocol,
    MatrixFreeSolverProtocol,
    PreconditionerProtocol,
    PreconditionerWithSetupProtocol,
)

__all__ = [
    "LinearSolverProtocol",
    "IterativeSolverProtocol",
    "MatrixFreeSolverProtocol",
    "PreconditionerProtocol",
    "PreconditionerWithSetupProtocol",
]
