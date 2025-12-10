"""
Core building blocks for Gaussian computations.

Provides shared computation for Gaussian log-PDF that can be reused
across distributions and likelihoods.
"""

from typing import Generic
import math

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.protocols import (
    SqrtCovarianceOperatorProtocol,
)


class GaussianLogPDFCore(Generic[Array]):
    """
    Shared computation for multivariate Gaussian log-PDF.

    Computes: log p(x | mean, Cov) = -0.5 * ||L^{-1}(x-mean)||^2 - log|L| - n/2*log(2*pi)

    where Cov = L @ L.T.

    Parameters
    ----------
    cov_op : SqrtCovarianceOperatorProtocol
        Covariance operator providing L and L^{-1} operations.
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _cov_op : SqrtCovarianceOperatorProtocol
        The covariance operator.
    _log_const : float
        Constant term: -n/2*log(2*pi) - log|L|
    """

    def __init__(
        self,
        cov_op: SqrtCovarianceOperatorProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._cov_op = cov_op
        self._nvars = cov_op.nvars()

        # Compute constant term: -n/2*log(2*pi) - log|L|
        self._log_const = (
            -0.5 * self._nvars * math.log(2 * math.pi)
            - cov_op.log_determinant()
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Get the dimension."""
        return self._nvars

    def covariance_operator(self) -> SqrtCovarianceOperatorProtocol[Array]:
        """Get the covariance operator."""
        return self._cov_op

    def compute(self, residuals: Array) -> Array:
        """
        Compute log-PDF given residuals (x - mean).

        Parameters
        ----------
        residuals : Array
            Residuals (x - mean). Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log-PDF values. Shape: (nsamples,)
        """
        # Whiten: L^{-1} @ residuals
        whitened = self._cov_op.apply_inv(residuals)

        # Squared Mahalanobis distance: sum of squared whitened values
        # ||L^{-1}(x-mean)||^2 = sum_i (L^{-1}(x-mean))_i^2
        sq_mahal = self._bkd.sum(whitened * whitened, axis=0)

        return self._log_const - 0.5 * sq_mahal

    def compute_gradient(self, residuals: Array) -> Array:
        """
        Compute gradient of log-PDF with respect to x.

        grad log p(x) = -Cov^{-1} @ (x - mean) = -L^{-T} @ L^{-1} @ residuals

        Parameters
        ----------
        residuals : Array
            Residuals (x - mean). Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Gradient values. Shape: (nvars, nsamples)
        """
        whitened = self._cov_op.apply_inv(residuals)
        return -self._cov_op.apply_inv_transpose(whitened)

    def log_normalization_constant(self) -> float:
        """
        Return the log normalization constant.

        Returns
        -------
        float
            -n/2*log(2*pi) - log|L|
        """
        return self._log_const
