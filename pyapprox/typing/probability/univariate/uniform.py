"""
Uniform univariate distribution.

Provides an analytically-defined Uniform distribution that implements
MarginalWithJacobianProtocol.
"""

from typing import Generic, Any, Tuple
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class UniformMarginal(Generic[Array]):
    """
    Uniform distribution on [lower, upper].

    Implements MarginalWithJacobianProtocol with analytical formulas for
    PDF, CDF, inverse CDF, and their Jacobians.

    The Uniform distribution has PDF:
        f(x) = 1 / (upper - lower)  for x in [lower, upper]
        f(x) = 0                     otherwise

    Parameters
    ----------
    lower : float
        Lower bound of the distribution.
    upper : float
        Upper bound of the distribution.
    bkd : Backend[Array]
        The backend to use for computations.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> dist = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> samples = np.array([0.1, 0.5, 0.9])
    >>> pdf_vals = dist(samples)  # All equal to 1.0
    >>> cdf_vals = dist.cdf(samples)  # Linear from 0 to 1
    """

    def __init__(self, lower: float, upper: float, bkd: Backend[Array]):
        if lower >= upper:
            raise ValueError(
                f"lower must be less than upper, got lower={lower}, upper={upper}"
            )

        self._bkd = bkd
        self._lower = float(lower)
        self._upper = float(upper)
        self._width = upper - lower
        self._pdf_val = 1.0 / self._width
        self._log_pdf_val = -math.log(self._width)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    @property
    def lower(self) -> float:
        """Return the lower bound."""
        return self._lower

    @property
    def upper(self) -> float:
        """Return the upper bound."""
        return self._upper

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
            The evaluated PDF values (constant within bounds).
        """
        return self._bkd.ones_like(samples) * self._pdf_val

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
            The evaluated log PDF values (constant within bounds).
        """
        return self._bkd.ones_like(samples) * self._log_pdf_val

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
        return (samples - self._lower) / self._width

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
            Quantile values in [lower, upper].
        """
        return self._lower + probs * self._width

    # Alias for compatibility
    ppf = invcdf

    def invcdf_jacobian(self, probs: Array) -> Array:
        """
        Compute Jacobian of inverse CDF.

        d(F^{-1})/dp = width (constant for uniform)

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1].

        Returns
        -------
        Array
            Jacobian values (all equal to width).
        """
        return self._bkd.ones_like(probs) * self._width

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
        usamples = np.random.uniform(self._lower, self._upper, nsamples)
        return self._bkd.reshape(self._bkd.asarray(usamples), (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        mean = (lower + upper) / 2

        Returns
        -------
        float
            Mean value.
        """
        return (self._lower + self._upper) / 2.0

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = (upper - lower)^2 / 12

        Returns
        -------
        float
            Variance value.
        """
        return self._width**2 / 12.0

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        return self._width / math.sqrt(12.0)

    def is_bounded(self) -> bool:
        """
        Check if the distribution is bounded.

        Returns
        -------
        bool
            True for Uniform (bounded on [lower, upper]).
        """
        return True

    def bounds(self) -> Tuple[float, float]:
        """
        Return the support bounds.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds.
        """
        return (self._lower, self._upper)

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
        return self.invcdf(self._bkd.asarray([eps, 1.0 - eps]))

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF.

        For uniform, d/dx log f(x) = 0 (constant PDF).

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        Array
            Jacobian values (all zeros). Shape: (1, nsamples)
        """
        return self._bkd.zeros((1, samples.shape[-1]))

    def pdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        For uniform, d/dx f(x) = 0 (constant PDF).

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        Array
            Jacobian values (all zeros). Shape: (1, nsamples)
        """
        return self._bkd.zeros((1, samples.shape[-1]))

    def __eq__(self, other: Any) -> bool:
        """Check equality with another UniformMarginal."""
        if not isinstance(other, UniformMarginal):
            return False
        return self._lower == other._lower and self._upper == other._upper

    def __repr__(self) -> str:
        """Return string representation."""
        return f"UniformMarginal(lower={self._lower}, upper={self._upper})"
