"""Linear system solvers for basis expansion fitting.

DEPRECATED: Solvers have moved to pyapprox.typing.optimization.linear.
This module re-exports for backward compatibility.

Import from the new location instead:
    from pyapprox.typing.optimization.linear import LeastSquaresSolver
"""

# Re-export from new location for backward compatibility
from pyapprox.typing.optimization.linear import (
    # Base
    LinearSystemSolver,
    SingleQoiSolverMixin,
    # Least squares
    LeastSquaresSolver,
    RidgeRegressionSolver,
    LinearlyConstrainedLstSqSolver,
    # Sparse
    OMPSolver,
    OMPTerminationFlag,
    BasisPursuitSolver,
    BasisPursuitDenoisingSolver,
    # Quantile
    QuantileRegressionSolver,
    ExpectileRegressionSolver,
)

__all__ = [
    # Base
    "LinearSystemSolver",
    "SingleQoiSolverMixin",
    # Least squares
    "LeastSquaresSolver",
    "RidgeRegressionSolver",
    "LinearlyConstrainedLstSqSolver",
    # Sparse
    "OMPSolver",
    "OMPTerminationFlag",
    "BasisPursuitSolver",
    "BasisPursuitDenoisingSolver",
    # Quantile
    "QuantileRegressionSolver",
    "ExpectileRegressionSolver",
]
