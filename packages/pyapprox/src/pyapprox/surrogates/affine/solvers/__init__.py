"""Linear system solvers for basis expansion fitting.

DEPRECATED: Solvers have moved to pyapprox.optimization.linear.
This module re-exports for backward compatibility.

Import from the new location instead:
    from pyapprox.optimization.linear import LeastSquaresSolver
"""

# Re-export from new location for backward compatibility
from pyapprox.optimization.linear import (
    BasisPursuitDenoisingSolver,
    BasisPursuitSolver,
    ExpectileRegressionSolver,
    # Least squares
    LeastSquaresSolver,
    LinearlyConstrainedLstSqSolver,
    # Base
    LinearSystemSolver,
    # Sparse
    OMPSolver,
    OMPTerminationFlag,
    # Quantile
    QuantileRegressionSolver,
    RidgeRegressionSolver,
    SingleQoiSolverMixin,
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
