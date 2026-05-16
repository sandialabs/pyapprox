"""PCA-based function encoder."""

from __future__ import annotations

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend


class PCAFunctionEncoder(Generic[Array]):
    """PCA-based function encoder via mean subtraction and SVD.

    Encode: codes = P.T @ (f - mean)
    Decode: f = P @ codes + mean
    Decode std: f_std = P @ std_codes  (no mean shift)

    Parameters
    ----------
    basis : Array, shape (ngrid, ncodes)
        Orthonormal basis columns (left singular vectors).
    mean : Array, shape (ngrid, 1)
        Mean of training data.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self, basis: Array, mean: Array, bkd: Backend[Array]
    ) -> None:
        self._basis = basis
        self._mean = mean
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ncodes(self) -> int:
        return self._basis.shape[1]

    def ngrid(self) -> int:
        return self._basis.shape[0]

    def basis(self) -> Array:
        """Return the (ngrid, ncodes) orthonormal basis."""
        return self._basis

    def mean(self) -> Array:
        """Return the (ngrid, 1) training-data mean."""
        return self._mean

    def encode(self, f_grid: Array) -> Array:
        """Encode grid values to codes. (ngrid, N) -> (ncodes, N)."""
        return self._bkd.dot(self._basis.T, f_grid - self._mean)

    def decode(self, codes: Array) -> Array:
        """Decode codes to grid values. (ncodes, N) -> (ngrid, N)."""
        return self._bkd.dot(self._basis, codes) + self._mean

    def decode_std(self, std_codes: Array) -> Array:
        """Propagate code-space std to grid space via variance propagation.

        sigma_v = sqrt(P^2 @ sigma_c^2), assuming uncorrelated code
        components. Exact for diagonal-kernel case (Gamma = S * I_m).
        Under-estimates posterior variance for coregionalization kernels
        which have non-zero cross-coordinate posterior covariance.
        TODO: proper handling requires predict_cov on latent regressor
        and decode_cov on encoders.
        """
        basis_sq = self._basis ** 2
        var_codes = std_codes ** 2
        var_grid = self._bkd.dot(basis_sq, var_codes)
        return self._bkd.sqrt(var_grid)

    @classmethod
    def fit_from_data(
        cls,
        f_grid_data: Array,
        bkd: Backend[Array],
        ncodes: Optional[int] = None,
        variance_fraction: Optional[float] = None,
    ) -> PCAFunctionEncoder[Array]:
        """Build encoder from training data using SVD.

        Parameters
        ----------
        f_grid_data : Array, shape (ngrid, N)
            Training function values on a grid.
        bkd : Backend[Array]
            Computational backend.
        ncodes : int, optional
            Number of codes (basis vectors) to keep.
        variance_fraction : float, optional
            Fraction of variance to retain (selects ncodes automatically).
            Exactly one of ncodes or variance_fraction must be provided.

        Returns
        -------
        PCAFunctionEncoder
            Fitted encoder.
        """
        if (ncodes is None) == (variance_fraction is None):
            raise ValueError(
                "Exactly one of ncodes or variance_fraction must be provided"
            )

        mean = bkd.mean(f_grid_data, axis=1)
        mean = bkd.reshape(mean, (mean.shape[0], 1))
        centered = f_grid_data - mean

        U, S, _Vh = bkd.svd(centered, full_matrices=False)

        if ncodes is not None:
            n = ncodes
        elif variance_fraction is not None:
            cumvar = bkd.cumsum(S * S)
            total_var = cumvar[-1]
            ratios = cumvar / total_var
            n = 1
            for ii in range(len(S)):
                ratio_val = float(bkd.to_numpy(ratios[ii]))
                if ratio_val >= variance_fraction:
                    n = ii + 1
                    break
            else:
                n = len(S)
        else:
            raise ValueError(
                "Exactly one of ncodes or variance_fraction "
                "must be provided"
            )

        basis = U[:, :n]
        return cls(basis, mean, bkd)
