"""
Operator-based multivariate Gaussian.

Provides a multivariate Gaussian using callback-based covariance operators,
suitable for infinite-dimensional or kernel-based covariances.
"""

from typing import Generic

import numpy as np

from pyapprox.probability.covariance import OperatorBasedCovarianceOperator
from pyapprox.probability.gaussian.core import GaussianLogPDFCore
from pyapprox.probability.protocols import SqrtCovarianceOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend


class OperatorBasedMultivariateGaussian(Generic[Array]):
    """
    Multivariate Gaussian with operator-based covariance.

    Uses callback functions for covariance operations, suitable for:
    - Infinite-dimensional fields (truncated to finite representation)
    - Kernel-based covariances
    - Implicit covariance operators

    Parameters
    ----------
    mean : Array
        Mean vector. Shape: (nvars, 1)
    cov_op : SqrtCovarianceOperatorProtocol
        Covariance operator providing apply, apply_inv, etc.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.covariance import (
    ...     OperatorBasedCovarianceOperator
    ... )
    >>> bkd = NumpyBkd()
    >>> # Define a scaling operator
    >>> scale = 2.0
    >>> cov_op = OperatorBasedCovarianceOperator(
    ...     apply_sqrt=lambda x: scale * x,
    ...     apply_sqrt_inv=lambda x: x / scale,
    ...     log_determinant=3 * np.log(scale),
    ...     nvars=3,
    ...     bkd=bkd,
    ... )
    >>> mean = np.zeros((3, 1))
    >>> dist = OperatorBasedMultivariateGaussian(mean, cov_op, bkd)
    """

    def __init__(
        self,
        mean: Array,
        cov_op: SqrtCovarianceOperatorProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._mean = mean
        self._nvars = mean.shape[0]

        # Validate mean shape
        if mean.shape != (self._nvars, 1):
            raise ValueError(f"mean must have shape (nvars, 1), got {mean.shape}")

        # Validate covariance operator dimension
        if cov_op.nvars() != self._nvars:
            raise ValueError(
                f"cov_op.nvars()={cov_op.nvars()} must match "
                f"mean.shape[0]={self._nvars}"
            )

        self._cov_op = cov_op

        # Create log-PDF core
        self._logpdf_core = GaussianLogPDFCore(cov_op, bkd)

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

    def covariance_operator(self) -> SqrtCovarianceOperatorProtocol[Array]:
        """
        Return the covariance operator.

        Returns
        -------
        SqrtCovarianceOperatorProtocol
            The covariance operator.
        """
        return self._cov_op

    def covariance_diagonal(
        self, batch_size: int = None, active_indices: Array = None
    ) -> Array:
        """
        Compute diagonal of covariance via probing.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for probe vectors.
        active_indices : Array, optional
            Indices of diagonal terms to compute.

        Returns
        -------
        Array
            Diagonal terms of covariance.
        """
        if isinstance(self._cov_op, OperatorBasedCovarianceOperator):
            return self._cov_op.compute_covariance_diagonal(
                batch_size=batch_size, active_indices=active_indices
            )

        # Fallback for other operator types
        if active_indices is None:
            active_indices = self._bkd.arange(self._nvars)

        n_active = active_indices.shape[0]
        if batch_size is None:
            batch_size = n_active

        diagonal = self._bkd.zeros((n_active,))
        cnt = 0

        while cnt < n_active:
            nvectors = min(batch_size, n_active - cnt)
            vectors = self._bkd.zeros((self._nvars, nvectors))
            for jj in range(nvectors):
                idx = self._bkd.to_int(active_indices[cnt + jj])
                vectors[idx, jj] = 1.0

            tmp = self._cov_op.apply_transpose(vectors)
            result = self._cov_op.apply(tmp)

            for jj in range(nvectors):
                idx = self._bkd.to_int(active_indices[cnt + jj])
                diagonal[cnt + jj] = result[idx, jj]

            cnt += nvectors

        return diagonal

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
        std_normal = self._bkd.asarray(np.random.normal(0, 1, (self._nvars, nsamples)))
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

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OperatorBasedMultivariateGaussian(nvars={self._nvars})"
