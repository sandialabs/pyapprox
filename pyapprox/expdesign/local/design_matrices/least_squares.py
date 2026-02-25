"""
Least squares design matrices.

For least squares regression, the design matrices are:
    M1_k = phi_k * phi_k^T  (outer product of basis functions)
    M0_k = sigma_k^2 * phi_k * phi_k^T  (weighted by noise variance)

For homoscedastic noise, M0 = M1.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend

from .base import DesignMatricesBase


class LeastSquaresDesignMatrices(DesignMatricesBase[Array], Generic[Array]):
    """
    Design matrices for least squares regression.

    Computes M0 and M1 matrices for ordinary least squares estimation.
    The information matrix M1 represents the Fisher information, and M0
    accounts for heteroscedastic noise.

    Parameters
    ----------
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    bkd : Backend[Array]
        Computational backend.
    noise_mult : Array, optional
        Noise multipliers at each design point. Shape: (ndesign_pts,)
        The noise variance at point k is proportional to noise_mult[k]^2.
        If None, assumes homoscedastic noise.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Linear model: [1, x] at points -1, 0, 1
    >>> design_factors = bkd.asarray([[1, -1], [1, 0], [1, 1]])
    >>> dm = LeastSquaresDesignMatrices(design_factors, bkd)
    >>> weights = bkd.asarray([[1/3], [1/3], [1/3]])
    >>> M1 = dm.M1(weights)  # Information matrix
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
        Compute individual design matrices for least squares.

        For least squares:
            M1_k = phi_k @ phi_k^T
            M0_k = sigma_k^2 * phi_k @ phi_k^T  (or M1_k if homoscedastic)

        Returns
        -------
        M0k : Array
            Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        M1k : Array
            Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        # M1_k = phi_k @ phi_k^T for each design point k
        # Using einsum: "ij,il->ijl" computes outer product for each row
        M1k = self._bkd.einsum("ij,il->ijl", self._design_factors, self._design_factors)

        if self.is_homoscedastic():
            # M0 = M1 when noise is constant
            return M1k, M1k

        # M0_k = sigma_k^2 * phi_k @ phi_k^T
        # noise_mult has shape (ndesign_pts,), need to broadcast
        M0k = self._bkd.einsum(
            "ij,il->ijl",
            self._design_factors,
            self._noise_mult[:, None] ** 2 * self._design_factors,
        )
        return M0k, M1k
