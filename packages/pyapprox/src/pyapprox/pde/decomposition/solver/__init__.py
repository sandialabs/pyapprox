"""DtN domain decomposition solver.

Provides:
- DtNResidual: Computes flux mismatch residual
- DtNJacobian: Computes Jacobian of residual
- DtNSolver: Newton solver for interface DOFs
"""

from pyapprox.pde.decomposition.solver.dtn_jacobian import (
    DtNJacobian,
    DtNJacobianExact,
    create_jacobian,
)
from pyapprox.pde.decomposition.solver.dtn_residual import (
    DtNResidual,
)
from pyapprox.pde.decomposition.solver.dtn_solver import (
    DtNSolver,
    DtNSolverResult,
    create_dtn_solver,
)

__all__ = [
    "DtNResidual",
    "DtNJacobian",
    "DtNJacobianExact",
    "create_jacobian",
    "DtNSolver",
    "DtNSolverResult",
    "create_dtn_solver",
]
