"""
R-optimal design criterion.

R-optimal designs minimize risk-based measures of prediction variance,
typically using Average Value at Risk (AVaR) or Conditional Value at Risk (CVaR).

The criterion computes the same prediction variances as G-optimal, but is
intended for use with AVaR-based optimization rather than minimax.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.local.protocols import DesignMatricesProtocol

from .g_optimal import GOptimalCriterion


class ROptimalCriterion(GOptimalCriterion[Array], Generic[Array]):
    """
    R-optimal design criterion.

    Minimizes risk-based measures (AVaR/CVaR) of prediction variance.
    Returns a vector of prediction variances, one for each prediction point,
    which is then processed by an AVaR optimizer.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    pred_factors : Array
        Basis function values at prediction points. Shape: (npred_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    R-optimal with beta=0 (risk-neutral) is equivalent to I-optimal
    (integrated prediction variance). As beta increases toward 1, the
    design becomes more conservative, focusing on worst-case scenarios.

    The objective returns a vector of prediction variances:
        J_j(w) = phi_j^T @ Cov(w) @ phi_j

    for each prediction point j. This is used with an AVaR optimizer.
    """

    pass


class ROptimalLeastSquaresCriterion(ROptimalCriterion[Array], Generic[Array]):
    """
    R-optimal criterion for least squares regression.

    Alias for ROptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass
