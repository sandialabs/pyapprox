"""
D-optimal design criterion.

D-optimal designs minimize the log determinant of the covariance matrix,
which is equivalent to maximizing the determinant of the information matrix.

For homoscedastic noise:
    objective = -log(det(M1))

For heteroscedastic noise:
    objective = log(det(M1^{-1} @ M0 @ M1^{-1}))
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.local.protocols import DesignMatricesProtocol

from .base import LocalOEDCriterionBase


class DOptimalCriterion(LocalOEDCriterionBase[Array], Generic[Array]):
    """
    D-optimal design criterion.

    Minimizes the log determinant of the parameter covariance matrix.
    This criterion has closed-form gradients for both homoscedastic
    and heteroscedastic noise models.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    For homoscedastic noise (M0 = M1):
        objective = -log(det(M1)) = log(det(M1^{-1}))

    For heteroscedastic noise:
        objective = log(det(M1^{-1} @ M0 @ M1^{-1}))

    The objective is formulated for minimization.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(design_matrices, bkd)

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate D-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Criterion value. Shape: (1, 1)
        """
        M1 = self._design_matrices.M1(design_weights)

        if self.is_homoscedastic():
            # log(det(M1_inv)) = -log(det(M1))
            # Use slogdet for numerical stability
            _, logdet_M1 = self._bkd.slogdet(M1)
            val = -logdet_M1
        else:
            M0 = self._design_matrices.M0(design_weights)
            M1_inv = self._bkd.inv(M1)
            gamma = M0 @ M1_inv
            # log(det(M1_inv @ gamma))
            _, logdet = self._bkd.slogdet(M1_inv @ gamma)
            val = logdet

        return self._bkd.reshape(val, (1, 1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of D-optimal criterion.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        if self.is_homoscedastic():
            return self._homoscedastic_jacobian(design_weights)
        return self._heteroscedastic_jacobian(design_weights)

    def _homoscedastic_jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian for homoscedastic noise.

        d/dw_k [-log(det(M1))] = -trace(M1^{-1} @ dM1/dw_k)
                                = -trace(M1^{-1} @ M1_k)
                                = -phi_k^T @ M1^{-1} @ phi_k
        """
        M1 = self._design_matrices.M1(design_weights)
        M1_inv = self._bkd.inv(M1)
        design_factors = self._design_matrices.design_factors()

        # For each design point k: -phi_k^T @ M1_inv @ phi_k
        # Using vectorized computation:
        # temp[j, k] = M1_inv[j, :] @ design_factors[k, :]
        # result[k] = sum_j design_factors[k, j] * temp[j, k]
        temp = design_factors.T * (M1_inv @ design_factors.T)
        jac = -self._bkd.sum(temp, axis=0)

        return self._bkd.reshape(jac, (1, self.nvars()))

    def _heteroscedastic_jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian for heteroscedastic noise (least squares).

        For f = log(det(M1^{-1} @ M0 @ M1^{-1})):
        df/dw_k = -2 * trace(M1^{-1} @ M1_k) + trace(M0^{-1} @ M0_k)
        """
        M1 = self._design_matrices.M1(design_weights)
        M1_inv = self._bkd.inv(M1)
        M0 = self._design_matrices.M0(design_weights)
        M0_inv = self._bkd.inv(M0)
        design_factors = self._design_matrices.design_factors()
        noise_mult = self._design_matrices.noise_mult()

        # Term 1: -2 * phi_k^T @ M1_inv @ phi_k
        temp1 = design_factors.T * (M1_inv @ design_factors.T)
        term1 = -2 * self._bkd.sum(temp1, axis=0)

        # Term 2: (sigma_k * phi_k)^T @ M0_inv @ (sigma_k * phi_k)
        # = sigma_k^2 * phi_k^T @ M0_inv @ phi_k
        weighted_factors = noise_mult[:, None] * design_factors
        temp2 = weighted_factors.T * (M0_inv @ weighted_factors.T)
        term2 = self._bkd.sum(temp2, axis=0)

        jac = term1 + term2
        return self._bkd.reshape(jac, (1, self.nvars()))


class DOptimalLeastSquaresCriterion(DOptimalCriterion[Array], Generic[Array]):
    """
    D-optimal criterion for least squares regression.

    Alias for DOptimalCriterion when used with LeastSquaresDesignMatrices.
    """

    pass


class DOptimalQuantileCriterion(DOptimalCriterion[Array], Generic[Array]):
    """
    D-optimal criterion for quantile regression.

    Uses the same formula as DOptimalCriterion but with different
    design matrices (QuantileDesignMatrices).

    The heteroscedastic Jacobian differs slightly due to the different
    structure of M0 and M1 in quantile regression.
    """

    def _heteroscedastic_jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian for heteroscedastic quantile regression.

        For quantile regression:
        M0_k = phi_k @ phi_k^T
        M1_k = (1/sigma_k) * phi_k @ phi_k^T

        df/dw_k = -2 * trace(M1^{-1} @ M1_k) + trace(M0^{-1} @ M0_k)
        """
        M1 = self._design_matrices.M1(design_weights)
        M1_inv = self._bkd.inv(M1)
        M0 = self._design_matrices.M0(design_weights)
        M0_inv = self._bkd.inv(M0)
        design_factors = self._design_matrices.design_factors()
        noise_mult = self._design_matrices.noise_mult()

        # Term 1: -2 * (1/sigma_k) * phi_k^T @ M1_inv @ phi_k
        weighted_factors = design_factors / noise_mult[:, None]
        temp1 = design_factors.T * (M1_inv @ weighted_factors.T)
        term1 = -2 * self._bkd.sum(temp1, axis=0)

        # Term 2: phi_k^T @ M0_inv @ phi_k
        temp2 = design_factors.T * (M0_inv @ design_factors.T)
        term2 = self._bkd.sum(temp2, axis=0)

        jac = term1 + term2
        return self._bkd.reshape(jac, (1, self.nvars()))
