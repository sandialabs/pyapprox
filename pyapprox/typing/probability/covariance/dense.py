"""
Dense Cholesky covariance operator.

Provides a covariance operator based on explicit Cholesky factorization
of a dense covariance matrix.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class DenseCholeskyCovarianceOperator(Generic[Array]):
    """
    Dense covariance operator using Cholesky factorization.

    Given covariance matrix Cov, computes L = chol(Cov) where Cov = L @ L.T,
    and provides efficient operations using L and L^{-1}.

    Parameters
    ----------
    covariance : Array
        Symmetric positive definite covariance matrix.
        Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _cov : Array
        Original covariance matrix.
    _cov_sqrt : Array
        Lower triangular Cholesky factor L.
    _cov_sqrt_inv : Array
        Inverse of Cholesky factor L^{-1}.
    _cov_inv : Array
        Inverse covariance (precision) matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBackend
    >>> bkd = NumpyBackend()
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> op = DenseCholeskyCovarianceOperator(cov, bkd)
    >>> x = np.array([[1.0], [0.0]])
    >>> Lx = op.apply(x)  # L @ x
    """

    def __init__(self, covariance: Array, bkd: Backend[Array]):
        self._bkd = bkd
        self._cov = covariance
        self._nvars = covariance.shape[0]

        # Compute Cholesky factorization
        self._cov_sqrt = bkd.cholesky(covariance)

        # Compute inverse of Cholesky factor via triangular solve
        identity = bkd.eye(self._nvars)
        self._cov_sqrt_inv = bkd.solve_triangular(
            self._cov_sqrt, identity, lower=True
        )

        # Compute inverse covariance: Cov^{-1} = L^{-T} @ L^{-1}
        self._cov_inv = self._cov_sqrt_inv.T @ self._cov_sqrt_inv

        # Compute log determinant: log|L| = sum(log(diag(L)))
        self._log_det = bkd.sum(bkd.log(bkd.diag(self._cov_sqrt)))

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
        Apply sqrt covariance: L @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._cov_sqrt @ vectors

    def apply_transpose(self, vectors: Array) -> Array:
        """
        Apply transpose of sqrt covariance: L.T @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L.T @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._cov_sqrt.T @ vectors

    def apply_inv(self, vectors: Array) -> Array:
        """
        Apply inverse of sqrt covariance: L^{-1} @ vectors.

        This is the whitening transformation.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L^{-1} @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._cov_sqrt_inv @ vectors

    def apply_inv_transpose(self, vectors: Array) -> Array:
        """
        Apply inverse transpose: L^{-T} @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L^{-T} @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._cov_sqrt_inv.T @ vectors

    def log_determinant(self) -> float:
        """
        Compute log determinant of L.

        Note: log|Cov| = 2 * log|L| since Cov = L @ L.T.

        Returns
        -------
        float
            log|L| = sum(log(diag(L)))
        """
        return float(self._log_det)

    def covariance(self) -> Array:
        """
        Return the full covariance matrix.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars)
        """
        return self._cov

    def covariance_inverse(self) -> Array:
        """
        Return the inverse covariance (precision) matrix.

        Returns
        -------
        Array
            Precision matrix. Shape: (nvars, nvars)
        """
        return self._cov_inv

    def cholesky_factor(self) -> Array:
        """
        Return the Cholesky factor L.

        Returns
        -------
        Array
            Lower triangular Cholesky factor. Shape: (nvars, nvars)
        """
        return self._cov_sqrt

    def cholesky_factor_inverse(self) -> Array:
        """
        Return the inverse Cholesky factor L^{-1}.

        Returns
        -------
        Array
            Inverse of Cholesky factor. Shape: (nvars, nvars)
        """
        return self._cov_sqrt_inv
