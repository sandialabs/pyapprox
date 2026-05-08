"""
Factory functions for local OED criteria and solvers.

These functions provide a convenient interface for creating OED components
without manually instantiating design matrices and criteria.
"""

from __future__ import annotations

from typing import Literal, Optional

from pyapprox.util.backends.protocols import Array, Backend

from .criteria import (
    AOptimalCriterion,
    COptimalCriterion,
    DOptimalCriterion,
    GOptimalCriterion,
    IOptimalCriterion,
    ROptimalCriterion,
)
from .criteria.base import LocalOEDCriterionBase
from .design_matrices import (
    LeastSquaresDesignMatrices,
    QuantileDesignMatrices,
)
from .design_matrices.base import DesignMatricesBase
from .solver import (
    AVaRLocalOEDSolver,
    MinimaxLocalOEDSolver,
    ScipyLocalOEDSolver,
)
from .solver.base import LocalOEDSolverBase

CriterionType = Literal["D", "A", "C", "I", "G", "R"]
RegressionType = Literal["least_squares", "quantile"]


def create_design_matrices(
    design_factors: Array,
    bkd: Backend[Array],
    regression_type: RegressionType = "least_squares",
    noise_mult: Optional[Array] = None,
    quantile: float = 0.5,
) -> DesignMatricesBase[Array]:
    """
    Create design matrices for OED.

    Parameters
    ----------
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    regression_type : str, optional
        Type of regression: "least_squares" or "quantile". Default: "least_squares"
    noise_mult : Array, optional
        Noise multipliers for heteroscedastic designs. Shape: (ndesign_pts,)
        If None, uses homoscedastic design.
    quantile : float, optional
        Quantile level for quantile regression. Default: 0.5 (median)

    Returns
    -------
    DesignMatricesBase
        Design matrices instance.
    """
    if regression_type == "least_squares":
        return LeastSquaresDesignMatrices(design_factors, bkd, noise_mult)
    elif regression_type == "quantile":
        return QuantileDesignMatrices(design_factors, bkd, quantile, noise_mult)
    else:
        raise ValueError(
            f"Unknown regression type: {regression_type}. "
            f"Expected 'least_squares' or 'quantile'."
        )


# TODO: Could we make this more extensible. Easier to add new
# criteria? without large if elif below
def create_criterion(
    criterion_type: CriterionType,
    design_factors: Array,
    bkd: Backend[Array],
    regression_type: RegressionType = "least_squares",
    noise_mult: Optional[Array] = None,
    quantile: float = 0.5,
    # Criterion-specific parameters
    vec: Optional[Array] = None,  # For C-optimal
    pred_factors: Optional[Array] = None,  # For I, G, R-optimal
    pred_weights: Optional[Array] = None,  # For I-optimal
) -> LocalOEDCriterionBase[Array]:
    """
    Create an OED criterion.

    Parameters
    ----------
    criterion_type : str
        Type of criterion: "D", "A", "C", "I", "G", or "R"
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    regression_type : str, optional
        Type of regression: "least_squares" or "quantile". Default: "least_squares"
    noise_mult : Array, optional
        Noise multipliers for heteroscedastic designs. Shape: (ndesign_pts,)
    quantile : float, optional
        Quantile level for quantile regression. Default: 0.5
    vec : Array, optional
        Linear combination vector for C-optimal. Shape: (ndesign_vars,)
    pred_factors : Array, optional
        Prediction factors for I, G, R-optimal. Shape: (npred_pts, ndesign_vars)
    pred_weights : Array, optional
        Prediction weights for I-optimal. Shape: (npred_pts,)

    Returns
    -------
    LocalOEDCriterionBase
        The criterion instance.
    """
    dm = create_design_matrices(
        design_factors, bkd, regression_type, noise_mult, quantile
    )

    if criterion_type == "D":
        return DOptimalCriterion(dm, bkd)

    elif criterion_type == "A":
        return AOptimalCriterion(dm, bkd)

    elif criterion_type == "C":
        if vec is None:
            raise ValueError("C-optimal criterion requires 'vec' parameter.")
        return COptimalCriterion(dm, vec, bkd)

    elif criterion_type == "I":
        if pred_factors is None:
            raise ValueError("I-optimal criterion requires 'pred_factors' parameter.")
        return IOptimalCriterion(dm, pred_factors, bkd, pred_weights)

    elif criterion_type == "G":
        if pred_factors is None:
            raise ValueError("G-optimal criterion requires 'pred_factors' parameter.")
        return GOptimalCriterion(dm, pred_factors, bkd)

    elif criterion_type == "R":
        if pred_factors is None:
            raise ValueError("R-optimal criterion requires 'pred_factors' parameter.")
        return ROptimalCriterion(dm, pred_factors, bkd)

    else:
        raise ValueError(
            f"Unknown criterion type: {criterion_type}. "
            f"Expected one of: 'D', 'A', 'C', 'I', 'G', 'R'."
        )


def create_solver(
    criterion_type: CriterionType,
    design_factors: Array,
    bkd: Backend[Array],
    regression_type: RegressionType = "least_squares",
    noise_mult: Optional[Array] = None,
    quantile: float = 0.5,
    # Criterion-specific parameters
    vec: Optional[Array] = None,
    pred_factors: Optional[Array] = None,
    pred_weights: Optional[Array] = None,
    # Solver-specific parameters
    alpha: float = 0.5,  # For R-optimal (AVaR)
    verbosity: int = 0,
    maxiter: Optional[int] = None,
) -> LocalOEDSolverBase[Array]:
    """
    Create an OED solver with the appropriate criterion.

    Parameters
    ----------
    criterion_type : str
        Type of criterion: "D", "A", "C", "I", "G", or "R"
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    regression_type : str, optional
        Type of regression: "least_squares" or "quantile". Default: "least_squares"
    noise_mult : Array, optional
        Noise multipliers for heteroscedastic designs.
    quantile : float, optional
        Quantile level for quantile regression. Default: 0.5
    vec : Array, optional
        Linear combination vector for C-optimal.
    pred_factors : Array, optional
        Prediction factors for I, G, R-optimal.
    pred_weights : Array, optional
        Prediction weights for I-optimal.
    alpha : float, optional
        Risk level for R-optimal (AVaR). Default: 0.5
    verbosity : int, optional
        Verbosity level for optimizer. Default: 0
    maxiter : int, optional
        Maximum iterations for optimizer.

    Returns
    -------
    LocalOEDSolverBase
        The solver instance.
    """
    criterion = create_criterion(
        criterion_type,
        design_factors,
        bkd,
        regression_type,
        noise_mult,
        quantile,
        vec,
        pred_factors,
        pred_weights,
    )

    # Choose appropriate solver based on criterion type
    if criterion_type in ("D", "A", "C", "I"):
        # Scalar criteria use standard scipy solver
        return ScipyLocalOEDSolver(criterion, bkd, verbosity=verbosity, maxiter=maxiter)
    elif criterion_type == "G":
        # G-optimal uses minimax solver
        return MinimaxLocalOEDSolver(
            criterion, bkd, verbosity=verbosity, maxiter=maxiter
        )
    elif criterion_type == "R":
        # R-optimal uses AVaR solver
        return AVaRLocalOEDSolver(
            criterion, bkd, alpha=alpha, verbosity=verbosity, maxiter=maxiter
        )
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")
