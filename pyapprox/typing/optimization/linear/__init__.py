"""Linear system solvers for fitting.

This module provides various solvers for finding coefficients
that minimize different objectives involving basis matrices.

Solver Categories
-----------------
Least Squares
    - LeastSquaresSolver: Standard least squares
    - RidgeRegressionSolver: L2 regularized (Tikhonov)
    - LinearlyConstrainedLstSqSolver: With equality constraints

Sparse
    - OMPSolver: Orthogonal Matching Pursuit
    - BasisPursuitSolver: L1 minimization
    - BasisPursuitDenoisingSolver: LASSO / L1 regularized

Quantile
    - QuantileRegressionSolver: Quantile regression via LP
    - ExpectileRegressionSolver: Expectile regression via IRLS
"""

from pyapprox.typing.optimization.linear.base import (
    LinearSystemSolver,
    SingleQoiSolverMixin,
)

from pyapprox.typing.optimization.linear.least_squares import (
    LeastSquaresSolver,
    RidgeRegressionSolver,
    LinearlyConstrainedLstSqSolver,
)

from pyapprox.typing.optimization.linear.sparse import (
    OMPSolver,
    OMPTerminationFlag,
    BasisPursuitSolver,
    BasisPursuitDenoisingSolver,
)

from pyapprox.typing.optimization.linear.quantile import (
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
