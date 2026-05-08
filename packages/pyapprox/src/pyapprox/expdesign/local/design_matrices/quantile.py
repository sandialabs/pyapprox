"""
Quantile regression design matrices.

For quantile regression, the design matrices differ from least squares:
    M0_k = phi_k * phi_k^T  (outer product of basis functions)
    M1_k = (1/sigma_k) * phi_k * phi_k^T  (weighted by inverse noise)

For homoscedastic noise, M0 = M1.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend

from .base import DesignMatricesBase


class QuantileDesignMatrices(DesignMatricesBase[Array], Generic[Array]):
    """
    Design matrices for quantile regression.

    Computes M0 and M1 matrices for quantile regression estimation.
    The matrices have a different structure than least squares due to
    the asymmetric loss function in quantile regression.

    Parameters
    ----------
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    noise_mult : Array, optional
        Noise multipliers at each design point. Shape: (ndesign_pts,)
        If None, assumes homoscedastic noise.

    Notes
    -----
    For quantile regression:
        M0_k = phi_k @ phi_k^T
        M1_k = (1/sigma_k) * phi_k @ phi_k^T

    This differs from least squares where noise_mult^2 appears in M0.
    """

    def __init__(
        self,
        design_factors: Array,
        bkd: Backend[Array],
        noise_mult: Optional[Array] = None,
    ) -> None:
        super().__init__(design_factors, bkd, noise_mult)

    def _compute_individual_design_matrices(self) -> Tuple[Array, Array]:
        """
        Compute individual design matrices for quantile regression.

        For quantile regression:
            M0_k = phi_k @ phi_k^T
            M1_k = (1/sigma_k) * phi_k @ phi_k^T  (or M0_k if homoscedastic)

        Returns
        -------
        M0k : Array
            Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        M1k : Array
            Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        # M0_k = phi_k @ phi_k^T for each design point k
        M0k = self._bkd.einsum("ij,il->ijl", self._design_factors, self._design_factors)

        if self.is_homoscedastic():
            # M0 = M1 when noise is constant
            return M0k, M0k

        # M1_k = (1/sigma_k) * phi_k @ phi_k^T
        M1k = self._bkd.einsum(
            "ij,il->ijl",
            self._design_factors,
            (1.0 / self._noise_mult[:, None]) * self._design_factors,
        )
        return M0k, M1k
