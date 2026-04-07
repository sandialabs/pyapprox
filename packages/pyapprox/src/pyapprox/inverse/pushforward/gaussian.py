"""
Gaussian pushforward through linear transformations.

A linear transformation applied to a Gaussian distribution results in
another Gaussian distribution. This module computes the mean and covariance
of the resulting distribution.
"""

from typing import Generic, Optional

from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.util.backends.protocols import Array, Backend


class GaussianPushforward(Generic[Array]):
    r"""
    Compute the mean and covariance of a Gaussian distribution when
    pushed forward through a linear model.

    A linear transformation applied to a Gaussian distribution results in
    another Gaussian distribution. This class computes the mean and covariance
    of the resulting Gaussian distribution after applying a linear
    transformation.

    Notes
    -----
    The transformation applied to the original Gaussian is:

    .. math::
        y = A z + b

    where:

    - ``z ~ N(mean, cov)`` is the original Gaussian distribution
    - ``A`` is the transformation matrix
    - ``b`` is a constant offset vector

    The resulting Gaussian distribution is:

    .. math::
        y ~ N(A mean + b, A cov A^T)

    Parameters
    ----------
    matrix : Array
        The matrix representing the linear transformation. Shape: (nqoi, nvars)
    mean : Array
        The mean of the original Gaussian distribution. Shape: (nvars, 1)
    covariance : Array
        The covariance of the original Gaussian distribution. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.
    offset : Array, optional
        A constant vector added to the linear transformation. Shape: (nqoi, 1)
        Defaults to zero vector.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> A = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2x2 transformation
    >>> mean = np.zeros((2, 1))
    >>> cov = np.eye(2)
    >>> pf = GaussianPushforward(A, mean, cov, bkd)
    >>> pf_mean = pf.mean()  # Shape: (2, 1)
    >>> pf_cov = pf.covariance()  # Shape: (2, 2)
    """

    def __init__(
        self,
        matrix: Array,
        mean: Array,
        covariance: Array,
        bkd: Backend[Array],
        offset: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._nqoi, self._nvars = matrix.shape
        self._matrix = matrix

        if mean.shape != (self._nvars, 1):
            raise ValueError(
                f"mean has wrong shape {mean.shape}, expected ({self._nvars}, 1)"
            )
        self._mean = mean

        if covariance.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"covariance has wrong shape {covariance.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._cov = covariance

        if offset is None:
            offset = bkd.zeros((self._nqoi, 1))
        if offset.shape != (self._nqoi, 1):
            raise ValueError(
                f"offset has wrong shape {offset.shape}, expected ({self._nqoi}, 1)"
            )
        self._offset = offset

        # Compute pushforward immediately
        self._compute()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest (output dimension)."""
        return self._nqoi

    def _compute(self) -> None:
        r"""
        Compute the mean and covariance of the resulting Gaussian distribution.

        The mean is computed as: ``mean = A @ x + b``
        The covariance is computed as: ``cov = A @ Sigma @ A^T``
        """
        self._pushforward_mean = self._matrix @ self._mean + self._offset
        self._pushforward_cov = self._matrix @ self._cov @ self._matrix.T

    def mean(self) -> Array:
        """
        Return the mean of the resulting Gaussian distribution.

        Returns
        -------
        Array
            Mean of the pushforward distribution. Shape: (nqoi, 1)
        """
        return self._pushforward_mean

    def covariance(self) -> Array:
        """
        Return the covariance of the resulting Gaussian distribution.

        Returns
        -------
        Array
            Covariance of the pushforward distribution. Shape: (nqoi, nqoi)
        """
        return self._pushforward_cov

    def pushforward_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Return the pushforward distribution as a Gaussian object.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            The resulting Gaussian distribution.
        """
        return DenseCholeskyMultivariateGaussian(
            self.mean(), self.covariance(), self._bkd
        )

    def __repr__(self) -> str:
        """Return string representation."""
        if self._nqoi > 1:
            return f"GaussianPushforward(nqoi={self._nqoi}, nvars={self._nvars})"
        mean_val = float(self._bkd.to_numpy(self._pushforward_mean)[0, 0])
        var_val = float(self._bkd.to_numpy(self._pushforward_cov)[0, 0])
        return (
            f"GaussianPushforward(nqoi={self._nqoi}, nvars={self._nvars}, "
            f"mean={mean_val:.4f}, var={var_val:.4f})"
        )
