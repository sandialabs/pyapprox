"""
Diagonal multivariate Gaussian.

Provides a multivariate Gaussian with diagonal covariance matrix
(independent components).
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.covariance import DiagonalCovarianceOperator
from pyapprox.typing.probability.gaussian.core import GaussianLogPDFCore


class DiagonalMultivariateGaussian(Generic[Array]):
    """
    Multivariate Gaussian with diagonal covariance.

    Efficient for high-dimensional independent Gaussian variables.
    All operations are O(n) instead of O(n^2) or O(n^3).

    Parameters
    ----------
    mean : Array
        Mean vector. Shape: (nvars, 1)
    variances : Array
        Variances (diagonal of covariance). Shape: (nvars,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mean = np.array([[0.0], [1.0], [2.0]])
    >>> variances = np.array([1.0, 4.0, 9.0])
    >>> dist = DiagonalMultivariateGaussian(mean, variances, bkd)
    >>> samples = dist.rvs(100)
    """

    def __init__(
        self, mean: Array, variances: Array, bkd: Backend[Array]
    ):
        self._bkd = bkd
        self._mean = mean
        self._nvars = mean.shape[0]

        # Validate mean shape
        if mean.shape != (self._nvars, 1):
            raise ValueError(
                f"mean must have shape (nvars, 1), got {mean.shape}"
            )

        # Validate variances shape
        if variances.shape != (self._nvars,):
            raise ValueError(
                f"variances must have shape ({self._nvars},), "
                f"got {variances.shape}"
            )

        # Create covariance operator
        self._cov_op = DiagonalCovarianceOperator(variances, bkd)

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

    def variances(self) -> Array:
        """
        Return the variances (diagonal of covariance).

        Returns
        -------
        Array
            Variances. Shape: (nvars,)
        """
        return self._cov_op.diagonal()

    def standard_deviations(self) -> Array:
        """
        Return the standard deviations.

        Returns
        -------
        Array
            Standard deviations. Shape: (nvars,)
        """
        return self._cov_op.standard_deviations()

    def covariance(self) -> Array:
        """
        Return the covariance matrix (diagonal).

        Returns
        -------
        Array
            Diagonal covariance matrix. Shape: (nvars, nvars)
        """
        return self._cov_op.covariance()

    def covariance_inverse(self) -> Array:
        """
        Return the inverse covariance (precision) matrix.

        Returns
        -------
        Array
            Diagonal precision matrix. Shape: (nvars, nvars)
        """
        return self._cov_op.covariance_inverse()

    def covariance_operator(self) -> DiagonalCovarianceOperator[Array]:
        """
        Return the covariance operator.

        Returns
        -------
        DiagonalCovarianceOperator
            The covariance operator.
        """
        return self._cov_op

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
        # Transform: sigma * z + mean
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

        For diagonal covariance:
            grad log p(x) = -(x - mean) / variances

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

    def kl_divergence(
        self, other: "DiagonalMultivariateGaussian"
    ) -> float:
        """
        Compute KL divergence KL(self || other).

        For diagonal Gaussians, this simplifies to sum over dimensions.

        Parameters
        ----------
        other : DiagonalMultivariateGaussian
            The other Gaussian distribution.

        Returns
        -------
        float
            KL divergence value.
        """
        var1 = self.variances()
        var2 = other.variances()
        mean_diff = other.mean() - self.mean()

        # KL = 0.5 * sum_i (var1_i/var2_i + (m2_i-m1_i)^2/var2_i - 1
        #                   + log(var2_i) - log(var1_i))
        trace_term = self._bkd.sum(var1 / var2)
        quad_term = self._bkd.sum(mean_diff[:, 0]**2 / var2)
        log_det_term = self._bkd.sum(self._bkd.log(var2)) - self._bkd.sum(
            self._bkd.log(var1)
        )

        return float(
            0.5 * (trace_term + quad_term - self._nvars + log_det_term)
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagonalMultivariateGaussian(nvars={self._nvars})"
