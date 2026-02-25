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

    >>> from pyapprox.expdesign.local import (
    ...     DOptimalCriterion,
    ...     LeastSquaresDesignMatrices,
    ...     ScipyLocalOEDSolver,
    ... )
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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

from .adjoint import (
    AdjointModel,
    LinearResidual,
    QuadraticFunctional,
)
from .criteria import (
    # A-optimal
    AOptimalCriterion,
    AOptimalLeastSquaresCriterion,
    AOptimalQuantileCriterion,
    # C-optimal
    COptimalCriterion,
    COptimalLeastSquaresCriterion,
    COptimalQuantileCriterion,
    # D-optimal
    DOptimalCriterion,
    DOptimalLeastSquaresCriterion,
    DOptimalQuantileCriterion,
    # G-optimal (minimax)
    GOptimalCriterion,
    GOptimalLeastSquaresCriterion,
    # I-optimal
    IOptimalCriterion,
    IOptimalLeastSquaresCriterion,
    LocalOEDCriterionBase,
    # R-optimal (AVaR)
    ROptimalCriterion,
    ROptimalLeastSquaresCriterion,
)
from .design_matrices import (
    DesignMatricesBase,
    LeastSquaresDesignMatrices,
    QuantileDesignMatrices,
)
from .factory import (
    create_criterion,
    create_design_matrices,
    create_solver,
)
from .protocols import (
    DesignMatricesProtocol,
    LocalOEDCriterionProtocol,
    LocalOEDCriterionWithHVPProtocol,
    LocalOEDSolverProtocol,
    OptimizerProtocol,
    OptimizerResultProtocol,
)
from .solver import (
    AVaRLocalOEDSolver,
    LocalOEDSolverBase,
    MinimaxLocalOEDSolver,
    ScipyLocalOEDSolver,
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
    # Criteria base
    "LocalOEDCriterionBase",
    # D-optimal
    "DOptimalCriterion",
    "DOptimalLeastSquaresCriterion",
    "DOptimalQuantileCriterion",
    # C-optimal
    "COptimalCriterion",
    "COptimalLeastSquaresCriterion",
    "COptimalQuantileCriterion",
    # A-optimal
    "AOptimalCriterion",
    "AOptimalLeastSquaresCriterion",
    "AOptimalQuantileCriterion",
    # I-optimal
    "IOptimalCriterion",
    "IOptimalLeastSquaresCriterion",
    # G-optimal (minimax)
    "GOptimalCriterion",
    "GOptimalLeastSquaresCriterion",
    # R-optimal (AVaR)
    "ROptimalCriterion",
    "ROptimalLeastSquaresCriterion",
    # Adjoint infrastructure
    "QuadraticFunctional",
    "LinearResidual",
    "AdjointModel",
    # Solvers
    "LocalOEDSolverBase",
    "ScipyLocalOEDSolver",
    "MinimaxLocalOEDSolver",
    "AVaRLocalOEDSolver",
    # Factory functions
    "create_design_matrices",
    "create_criterion",
    "create_solver",
]
