"""
Dense Cholesky multivariate Gaussian.

Provides a multivariate Gaussian with dense covariance matrix using
Cholesky factorization.
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.typing.probability.gaussian.core import GaussianLogPDFCore


class DenseCholeskyMultivariateGaussian(Generic[Array]):
    """
    Multivariate Gaussian with dense Cholesky covariance.

    Uses explicit Cholesky factorization for efficient operations on
    moderate-dimensional problems.

    Parameters
    ----------
    mean : Array
        Mean vector. Shape: (nvars, 1)
    covariance : Array
        Covariance matrix. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mean = np.array([[0.0], [0.0]])
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> dist = DenseCholeskyMultivariateGaussian(mean, cov, bkd)
    >>> samples = dist.rvs(100)
    """

    def __init__(
        self, mean: Array, covariance: Array, bkd: Backend[Array]
    ):
        self._bkd = bkd
        self._mean = mean
        self._nvars = mean.shape[0]

        # Validate mean shape
        if mean.shape != (self._nvars, 1):
            raise ValueError(
                f"mean must have shape (nvars, 1), got {mean.shape}"
            )

        # Validate covariance shape
        if covariance.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"covariance must have shape ({self._nvars}, {self._nvars}), "
                f"got {covariance.shape}"
            )

        # Create covariance operator
        self._cov_op = DenseCholeskyCovarianceOperator(covariance, bkd)

        # Create log-PDF core
        self._logpdf_core = GaussianLogPDFCore(self._cov_op, bkd)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def mean(self) -> Array:
        """
        Return the mean vector.

        Returns
        -------
        Array
            Mean vector. Shape: (nvars, 1)
        """
        return self._mean

    def covariance(self) -> Array:
        """
        Return the covariance matrix.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars)
        """
        return self._cov_op.covariance()

    def covariance_inverse(self) -> Array:
        """
        Return the inverse covariance (precision) matrix.

        Returns
        -------
        Array
            Precision matrix. Shape: (nvars, nvars)
        """
        return self._cov_op.covariance_inverse()

    def covariance_operator(self) -> DenseCholeskyCovarianceOperator[Array]:
        """
        Return the covariance operator.

        Returns
        -------
        DenseCholeskyCovarianceOperator
            The covariance operator.
        """
        return self._cov_op

    def cholesky_factor(self) -> Array:
        """
        Return the Cholesky factor L where Cov = L @ L.T.

        Returns
        -------
        Array
            Lower triangular Cholesky factor. Shape: (nvars, nvars)
        """
        return self._cov_op.cholesky_factor()

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        # Generate standard normal samples
        std_normal = self._bkd.asarray(
            np.random.normal(0, 1, (self._nvars, nsamples))
        )
        # Transform: L @ z + mean
        return self._cov_op.apply(std_normal) + self._mean

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (nsamples,)
        """
        residuals = samples - self._mean
        return self._logpdf_core.compute(residuals)

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            PDF values. Shape: (nsamples,)
        """
        return self._bkd.exp(self.logpdf(samples))

    def logpdf_gradient(self, samples: Array) -> Array:
        """
        Compute gradient of log-PDF with respect to samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Gradient values. Shape: (nvars, nsamples)
        """
        residuals = samples - self._mean
        return self._logpdf_core.compute_gradient(residuals)

    def logpdf_hessian(self) -> Array:
        """
        Return the Hessian of log-PDF (constant for Gaussian).

        Returns
        -------
        Array
            Hessian matrix (negative precision). Shape: (nvars, nvars)
        """
        return -self._cov_op.covariance_inverse()

    def kl_divergence(
        self, other: "DenseCholeskyMultivariateGaussian"
    ) -> float:
        """
        Compute KL divergence KL(self || other).

        KL( N(m1, C1) || N(m2, C2) ) =
            0.5 * (tr(C2^{-1} C1) + (m2-m1)^T C2^{-1} (m2-m1)
                   - n + log|C2| - log|C1|)

        Parameters
        ----------
        other : DenseCholeskyMultivariateGaussian
            The other Gaussian distribution.

        Returns
        -------
        float
            KL divergence value.
        """
        cov1 = self.covariance()
        cov2 = other.covariance()
        cov2_inv = other.covariance_inverse()

        mean_diff = other.mean() - self.mean()

        # tr(C2^{-1} C1)
        trace_term = self._bkd.sum(cov2_inv * cov1)

        # (m2-m1)^T C2^{-1} (m2-m1)
        quad_term = float(
            (mean_diff.T @ (cov2_inv @ mean_diff)).squeeze()
        )

        # log|C2| - log|C1| = 2*(log|L2| - log|L1|)
        log_det_term = 2.0 * (
            other.covariance_operator().log_determinant()
            - self._cov_op.log_determinant()
        )

        return 0.5 * (trace_term + quad_term - self._nvars + log_det_term)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DenseCholeskyMultivariateGaussian(nvars={self._nvars})"
