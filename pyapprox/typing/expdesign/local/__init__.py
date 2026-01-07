"""
Local optimal experimental design module.

This module provides functionality for computing locally optimal experimental
designs for linear regression models. Local OED criteria include:

- D-optimal: Minimize log det of covariance matrix
- A-optimal: Minimize trace of covariance matrix
- C-optimal: Minimize variance of linear combination
- I-optimal: Minimize integrated prediction variance
- G-optimal: Minimize maximum prediction variance
- R-optimal: Minimize risk-based (AVaR) prediction variance

Example
-------
Basic D-optimal design:

    >>> from pyapprox.typing.expdesign.local import (
    ...     DOptimalCriterion,
    ...     LeastSquaresDesignMatrices,
    ...     ScipyLocalOEDSolver,
    ... )
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> # Design factors: basis function values at design points
    >>> # Shape: (ndesign_pts, ndesign_vars)
    >>> design_factors = bkd.asarray([[1, -1], [1, 0], [1, 1]])
    >>>
    >>> # Create design matrices
    >>> design_matrices = LeastSquaresDesignMatrices(design_factors, bkd)
    >>>
    >>> # Create criterion
    >>> criterion = DOptimalCriterion(design_matrices, bkd)
    >>>
    >>> # Create solver and find optimal design
    >>> solver = ScipyLocalOEDSolver(criterion, bkd)
    >>> optimal_weights = solver.construct()
"""

from .protocols import (
    LocalOEDCriterionProtocol,
    LocalOEDCriterionWithHVPProtocol,
    DesignMatricesProtocol,
    OptimizerResultProtocol,
    OptimizerProtocol,
    LocalOEDSolverProtocol,
)
from .design_matrices import (
    DesignMatricesBase,
    LeastSquaresDesignMatrices,
    QuantileDesignMatrices,
)
from .criteria import (
    LocalOEDCriterionBase,
    DOptimalCriterion,
    DOptimalLeastSquaresCriterion,
    DOptimalQuantileCriterion,
)
from .adjoint import (
    QuadraticFunctional,
    LinearResidual,
    AdjointModel,
)

__all__ = [
    # Protocols
    "LocalOEDCriterionProtocol",
    "LocalOEDCriterionWithHVPProtocol",
    "DesignMatricesProtocol",
    "OptimizerResultProtocol",
    "OptimizerProtocol",
    "LocalOEDSolverProtocol",
    # Design matrices
    "DesignMatricesBase",
    "LeastSquaresDesignMatrices",
    "QuantileDesignMatrices",
    # Criteria
    "LocalOEDCriterionBase",
    "DOptimalCriterion",
    "DOptimalLeastSquaresCriterion",
    "DOptimalQuantileCriterion",
    # Adjoint infrastructure
    "QuadraticFunctional",
    "LinearResidual",
    "AdjointModel",
]
