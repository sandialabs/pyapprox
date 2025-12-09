"""
Diagonal covariance operator.

Provides an efficient covariance operator for diagonal covariance matrices
where all operations reduce to elementwise operations.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class DiagonalCovarianceOperator(Generic[Array]):
    """
    Diagonal covariance operator.

    For diagonal Cov = diag(sigma^2), the Cholesky factor is L = diag(sigma),
    and all operations become efficient elementwise operations.

    Parameters
    ----------
    variances : Array
        Diagonal of the covariance matrix (variances).
        Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _variances : Array
        Variances (diagonal of covariance).
    _std : Array
        Standard deviations (sqrt of variances).
    _std_inv : Array
        Inverse standard deviations.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBackend
    >>> bkd = NumpyBackend()
    >>> variances = np.array([1.0, 4.0, 9.0])
    >>> op = DiagonalCovarianceOperator(variances, bkd)
    >>> x = np.array([[1.0], [1.0], [1.0]])
    >>> Lx = op.apply(x)  # [1, 2, 3]
    """

    def __init__(self, variances: Array, bkd: Backend[Array]):
        if variances.ndim != 1:
            raise ValueError("variances must be a 1D array")

        self._bkd = bkd
        self._variances = variances
        self._nvars = variances.shape[0]

        # Compute standard deviations (Cholesky diagonal)
        self._std = bkd.sqrt(variances)
        self._std_inv = 1.0 / self._std

        # Log determinant: log|L| = sum(log(sigma))
        self._log_det = bkd.sum(bkd.log(self._std))

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the dimension of the covariance matrix."""
        return self._nvars

    def _check_vectors(self, vectors: Array) -> None:
        """Validate input vectors shape."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[0] != self._nvars:
            raise ValueError(
                f"vectors had {vectors.shape[0]} rows, expected {self._nvars}"
            )

    def apply(self, vectors: Array) -> Array:
        """
        Apply sqrt covariance: L @ vectors = diag(sigma) @ vectors.

        For diagonal L, this is elementwise multiplication.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            sigma * vectors (elementwise). Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._std[:, None] * vectors

    def apply_transpose(self, vectors: Array) -> Array:
        """
        Apply transpose of sqrt covariance: L.T @ vectors.

        For diagonal L, L.T = L, so this equals apply().

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            sigma * vectors (elementwise). Shape: (nvars, nvectors)
        """
        # For diagonal matrix, transpose equals itself
        return self.apply(vectors)

    def apply_inv(self, vectors: Array) -> Array:
        """
        Apply inverse of sqrt covariance: L^{-1} @ vectors.

        For diagonal L, this is elementwise division.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            vectors / sigma (elementwise). Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._std_inv[:, None] * vectors

    def apply_inv_transpose(self, vectors: Array) -> Array:
        """
        Apply inverse transpose: L^{-T} @ vectors.

        For diagonal L, L^{-T} = L^{-1}, so this equals apply_inv().

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            vectors / sigma (elementwise). Shape: (nvars, nvectors)
        """
        # For diagonal matrix, inverse transpose equals inverse
        return self.apply_inv(vectors)

    def log_determinant(self) -> float:
        """
        Compute log determinant of L.

        For diagonal L = diag(sigma), log|L| = sum(log(sigma)).

        Returns
        -------
        float
            log|L| = sum(log(sigma))
        """
        return float(self._log_det)

    def diagonal(self) -> Array:
        """
        Return the diagonal of the covariance matrix.

        Returns
        -------
        Array
            Variances. Shape: (nvars,)
        """
        return self._variances

    def covariance(self) -> Array:
        """
        Return the full covariance matrix (diagonal).

        Returns
        -------
        Array
            Diagonal covariance matrix. Shape: (nvars, nvars)
        """
        return self._bkd.diag(self._variances)

    def covariance_inverse(self) -> Array:
        """
        Return the inverse covariance (precision) matrix.

        Returns
        -------
        Array
            Diagonal precision matrix. Shape: (nvars, nvars)
        """
        return self._bkd.diag(1.0 / self._variances)

    def standard_deviations(self) -> Array:
        """
        Return the standard deviations (sqrt of variances).

        Returns
        -------
        Array
            Standard deviations. Shape: (nvars,)
        """
        return self._std
