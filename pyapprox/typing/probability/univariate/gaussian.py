"""
Gaussian (normal) univariate distribution.

Provides an analytically-defined Gaussian distribution that implements
MarginalWithJacobianProtocol.
"""

from typing import Generic, Any
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class GaussianMarginal(Generic[Array]):
    """
    Gaussian (normal) distribution.

    Implements MarginalWithJacobianProtocol with analytical formulas for
    PDF, CDF, inverse CDF, and their Jacobians.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    stdev : float
        Standard deviation of the Gaussian distribution.
    bkd : Backend[Array]
        The backend to use for computations.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> dist = GaussianMarginal(mean=0.0, stdev=1.0, bkd=bkd)
    >>> samples = np.array([0.0, 1.0, -1.0])
    >>> pdf_vals = dist(samples)  # PDF values
    >>> cdf_vals = dist.cdf(samples)  # CDF values
    """

    def __init__(self, mean: float, stdev: float, bkd: Backend[Array]):
        self._bkd = bkd
        self._mean = float(mean)
        self._stdev = float(stdev)
        self._var = stdev**2
        self._log_const = -0.5 * math.log(2.0 * math.pi) - math.log(stdev)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF.

        Returns
        -------
        Array
            The evaluated PDF values.
        """
        return self._bkd.exp(self.logpdf(samples))

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PDF.

        Returns
        -------
        Array
            The evaluated log PDF values.
        """
        z = (samples - self._mean) / self._stdev
        return self._log_const - 0.5 * z**2

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF.

        Returns
        -------
        Array
            CDF values in [0, 1].
        """
        z = (samples - self._mean) / (self._stdev * math.sqrt(2.0))
        return 0.5 * (1.0 + self._bkd.erf(z))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1].

        Returns
        -------
        Array
            Quantile values.
        """
        return (
            math.sqrt(2.0) * self._bkd.erfinv(2.0 * probs - 1.0)
        ) * self._stdev + self._mean

    # Alias for compatibility
    ppf = invcdf

    def invcdf_jacobian(self, probs: Array) -> Array:
        """
        Compute Jacobian of inverse CDF.

        d(F^{-1})/dp = 1 / pdf(F^{-1}(p))

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1].

        Returns
        -------
        Array
            Jacobian values.
        """
        samples = self.invcdf(probs)
        pdf_vals = self(samples)
        return 1.0 / pdf_vals

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
            Random samples. Shape: (1, nsamples) for protocol compliance.
        """
        usamples = self._bkd.asarray(np.random.uniform(0, 1, nsamples))
        samples = self.invcdf(usamples)
        return self._bkd.reshape(samples, (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        Returns
        -------
        float
            Mean value.
        """
        return self._mean

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        Returns
        -------
        float
            Variance value.
        """
        return self._var

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        return self._stdev

    def is_bounded(self) -> bool:
        """
        Check if the distribution is bounded.

        Returns
        -------
        bool
            False for Gaussian (unbounded support).
        """
        return False

    def interval(self, alpha: float) -> Array:
        """
        Compute the interval with given probability content.

        Parameters
        ----------
        alpha : float
            Probability content of the interval (0 < alpha < 1).

        Returns
        -------
        Array
            Interval [lower, upper] such that P(lower < X < upper) = alpha.
        """
        eps = (1.0 - alpha) / 2.0
        return self.invcdf(self._bkd.array([eps, 1.0 - eps]))

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)
        """
        grad = -(samples - self._mean) / self._var
        return self._bkd.reshape(grad, (1, -1))

    def pdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)
        """
        pdf_vals = self(samples)
        logpdf_jac = self.logpdf_jacobian(samples)
        return pdf_vals * logpdf_jac

    def __eq__(self, other: Any) -> bool:
        """Check equality with another GaussianMarginal."""
        if not isinstance(other, GaussianMarginal):
            return False
        return self._mean == other._mean and self._stdev == other._stdev

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianMarginal(mean={self._mean}, stdev={self._stdev})"
