"""
Protocols for covariance operators.

These protocols define the interface for covariance operators used in
multivariate Gaussians and likelihood computations.

The key abstraction is the square-root (Cholesky) factorization:
    Cov = L @ L.T

where L is lower triangular. All operations are defined in terms of L.

Protocol Hierarchy
------------------
SqrtCovarianceOperatorProtocol
    Core protocol with apply, inverse, transpose operations.
CovarianceOperatorProtocol
    Extends with full covariance access (dense storage).
DiagonalCovarianceOperatorProtocol
    Specialized for diagonal covariance matrices.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SqrtCovarianceOperatorProtocol(Protocol, Generic[Array]):
    """
    Protocol for square-root covariance operators.

    The operator represents L where Cov = L @ L.T.
    All operations are defined in terms of L (lower triangular Cholesky factor).

    This is the primary interface for covariance operations, enabling:
    - Sampling: z ~ N(0,I), then L @ z ~ N(0, Cov)
    - Log-PDF computation via log|L| and L^{-1} @ (x - mean)
    - Whitening: L^{-1} @ x

    Methods
    -------
    apply(vectors)
        Compute L @ vectors.
    apply_transpose(vectors)
        Compute L.T @ vectors.
    apply_inv(vectors)
        Compute L^{-1} @ vectors.
    apply_inv_transpose(vectors)
        Compute L^{-T} @ vectors.
    nvars()
        Dimension of the covariance.
    log_determinant()
        Compute log|L| = 0.5 * log|Cov|.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the dimension of the covariance matrix.

        Returns
        -------
        int
            Number of variables.
        """
        ...

    def apply(self, vectors: Array) -> Array:
        """
        Apply the sqrt covariance: L @ vectors.

        If vectors are standard normal samples, the result is
        samples from N(0, Cov).

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L @ vectors. Shape: (nvars, nvectors)
        """
        ...

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
        ...

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
        ...

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
        ...

    def log_determinant(self) -> float:
        """
        Compute log determinant of L.

        Note: log|Cov| = 2 * log|L| since Cov = L @ L.T.

        Returns
        -------
        float
            log|L| = sum(log(diag(L)))
        """
        ...


@runtime_checkable
class CovarianceOperatorProtocol(Protocol, Generic[Array]):
    """
    Covariance operator with full matrix access.

    Extends SqrtCovarianceOperatorProtocol with methods to access
    the full covariance and its inverse (requires dense storage).

    Methods
    -------
    covariance()
        Return the full covariance matrix.
    covariance_inverse()
        Return the inverse covariance matrix.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def apply(self, vectors: Array) -> Array: ...

    def apply_transpose(self, vectors: Array) -> Array: ...

    def apply_inv(self, vectors: Array) -> Array: ...

    def apply_inv_transpose(self, vectors: Array) -> Array: ...

    def log_determinant(self) -> float: ...

    def covariance(self) -> Array:
        """
        Return the full covariance matrix.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars)
        """
        ...

    def covariance_inverse(self) -> Array:
        """
        Return the inverse covariance (precision) matrix.

        Returns
        -------
        Array
            Precision matrix. Shape: (nvars, nvars)
        """
        ...


@runtime_checkable
class DiagonalCovarianceOperatorProtocol(Protocol, Generic[Array]):
    """
    Specialized protocol for diagonal covariance matrices.

    For diagonal Cov = diag(sigma^2), operations are elementwise:
    - L = diag(sigma)
    - L^{-1} = diag(1/sigma)

    Methods
    -------
    diagonal()
        Return the diagonal of the covariance.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def apply(self, vectors: Array) -> Array: ...

    def apply_transpose(self, vectors: Array) -> Array: ...

    def apply_inv(self, vectors: Array) -> Array: ...

    def apply_inv_transpose(self, vectors: Array) -> Array: ...

    def log_determinant(self) -> float: ...

    def diagonal(self) -> Array:
        """
        Return the diagonal of the covariance matrix.

        Returns
        -------
        Array
            Variances. Shape: (nvars,)
        """
        ...
