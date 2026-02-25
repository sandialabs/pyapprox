"""
G-optimal design criterion.

G-optimal designs minimize the maximum prediction variance:
    J(w) = max_j phi_j^T @ Cov(w) @ phi_j

This is formulated as a minimax problem where each prediction point
contributes one objective value.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.local.protocols import DesignMatricesProtocol
from pyapprox.expdesign.local.adjoint import AdjointModel

from .base import LocalOEDCriterionBase


class GOptimalCriterion(LocalOEDCriterionBase[Array], Generic[Array]):
    """
    G-optimal design criterion.

    Minimizes the maximum prediction variance over prediction points.
    Returns a vector of prediction variances, one for each prediction point.

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
    The objective returns a vector of prediction variances:
        J_j(w) = phi_j^T @ Cov(w) @ phi_j

    for each prediction point j. This is used with a minimax optimizer
    to minimize max_j J_j(w).

    Unlike I-optimal which integrates/averages, G-optimal returns individual
    values for minimax optimization.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        pred_factors: Array,
        bkd: Backend[Array],
    ) -> None:
        super().__init__(design_matrices, bkd)
        self._pred_factors = pred_factors
        self._npred_pts = pred_factors.shape[0]
        self._setup_adjoints()

    def _setup_adjoints(self) -> None:
        """Set up adjoint models for each prediction point."""
        self._adjoints = []
        for j in range(self._npred_pts):
            adjoint = AdjointModel(self._design_matrices, self._bkd)
            # Each prediction point phi_j is a vector
            vec = self._pred_factors[j, :]
            adjoint.set_vector(vec)
            self._adjoints.append(adjoint)

    def nqoi(self) -> int:
        """Number of quantities of interest (prediction points)."""
        return self._npred_pts

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate G-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Vector of prediction variances. Shape: (npred_pts, 1)
        """
        vals = []
        for adjoint in self._adjoints:
            val = adjoint.value(design_weights)  # Shape: (1, 1)
            vals.append(val[0, 0])
        return self._bkd.reshape(self._bkd.stack(vals), (self._npred_pts, 1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of G-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (npred_pts, nvars)
        """
        jacs = []
        for adjoint in self._adjoints:
            jac = adjoint.jacobian(design_weights)  # Shape: (1, nvars)
            jacs.append(jac[0, :])
        return self._bkd.stack(jacs)

    # NOTE: hvp is NOT implemented for G-optimal because it's a multi-output
    # objective used with minimax optimization. Following the optional methods
    # convention, hvp is simply not defined on this class.


class GOptimalLeastSquaresCriterion(GOptimalCriterion[Array], Generic[Array]):
    """
    G-optimal criterion for least squares regression.

    Alias for GOptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass
