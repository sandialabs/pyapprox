"""
Uniform univariate distribution.

Provides an analytically-defined Uniform distribution that implements
MarginalWithJacobianProtocol.
"""

from typing import Generic, Any, Tuple
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)


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
    >>> samples = np.array([[0.1, 0.5, 0.9]])  # Shape: (1, 3)
    >>> pdf_vals = dist(samples)  # All equal to 1.0, shape: (1, 3)
    >>> cdf_vals = dist.cdf(samples)  # Linear from 0 to 1, shape: (1, 3)
    """

    def __init__(self, lower: float, upper: float, bkd: Backend[Array]):
        if lower >= upper:
            raise ValueError(
                f"lower must be less than upper, got lower={lower}, upper={upper}"
            )

        self._bkd = bkd

        # Create hyperparameter list for parameter optimization
        # Both bounds are unbounded real numbers (no log transform)
        self._lower_hyp = HyperParameter(
            name="lower",
            nparams=1,
            values=lower,
            bounds=(-1e10, 1e10),  # Effectively unbounded
            bkd=bkd,
        )
        self._upper_hyp = HyperParameter(
            name="upper",
            nparams=1,
            values=upper,
            bounds=(-1e10, 1e10),  # Effectively unbounded
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._lower_hyp, self._upper_hyp])

    def _validate_input(self, samples: Array) -> Array:
        """Validate that input is 2D with shape (1, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (1, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Univariate distribution expects shape (1, nsamples), "
                f"got {samples.shape}"
            )
        return samples[0]  # Return 1D for internal computation

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for parameter optimization."""
        return self._hyp_list

    def nparams(self) -> int:
        """Return the number of distribution parameters (lower and upper)."""
        return self._hyp_list.nparams()

    def _get_lower(self) -> Array:
        """Get lower bound as array (preserves autograd graph)."""
        return self._lower_hyp.get_values()[0]

    def _get_upper(self) -> Array:
        """Get upper bound as array (preserves autograd graph)."""
        return self._upper_hyp.get_values()[0]

    def _get_width(self) -> Array:
        """Get width as array (preserves autograd graph)."""
        return self._get_upper() - self._get_lower()

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    def lower(self) -> float:
        """Return the lower bound."""
        return float(self._bkd.to_numpy(self._lower_hyp.get_values())[0])

    def upper(self) -> float:
        """Return the upper bound."""
        return float(self._bkd.to_numpy(self._upper_hyp.get_values())[0])

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            The evaluated PDF values (constant within bounds). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        width = self._get_width()
        pdf_val = 1.0 / width
        result = self._bkd.ones_like(samples_1d) * pdf_val
        return self._bkd.reshape(result, (1, -1))

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            The evaluated log PDF values (constant within bounds). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        width = self._get_width()
        log_pdf_val = -self._bkd.log(width)
        result = self._bkd.ones_like(samples_1d) * log_pdf_val
        return self._bkd.reshape(result, (1, -1))

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        lower = self._get_lower()
        width = self._get_width()
        result = (samples_1d - lower) / width
        return self._bkd.reshape(result, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Quantile values in [lower, upper]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        lower = self._get_lower()
        width = self._get_width()
        result = lower + probs_1d * width
        return self._bkd.reshape(result, (1, -1))

    # Alias for compatibility
    ppf = invcdf

    def invcdf_jacobian(self, probs: Array) -> Array:
        """
        Compute Jacobian of inverse CDF.

        d(F^{-1})/dp = width (constant for uniform)

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values (all equal to width). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        width = self._get_width()
        result = self._bkd.ones_like(probs_1d) * width
        return self._bkd.reshape(result, (1, -1))

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
            Random samples. Shape: (1, nsamples)
        """
        lower_val = self.lower()
        upper_val = self.upper()
        usamples = np.random.uniform(lower_val, upper_val, nsamples)
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
        return (self.lower() + self.upper()) / 2.0

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = (upper - lower)^2 / 12

        Returns
        -------
        float
            Variance value.
        """
        width = self.upper() - self.lower()
        return width**2 / 12.0

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        width = self.upper() - self.lower()
        return width / math.sqrt(12.0)

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
        return (self.lower(), self.upper())

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
            Shape: (1, 2)
        """
        eps = (1.0 - alpha) / 2.0
        probs_2d = self._bkd.array([[eps, 1.0 - eps]])  # Shape: (1, 2)
        return self.invcdf(probs_2d)

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF.

        For uniform, d/dx log f(x) = 0 (constant PDF).

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values (all zeros). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        return self._bkd.zeros((1, len(samples_1d)))

    def pdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        For uniform, d/dx f(x) = 0 (constant PDF).

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values (all zeros). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        return self._bkd.zeros((1, len(samples_1d)))

    def __eq__(self, other: Any) -> bool:
        """Check equality with another UniformMarginal."""
        if not isinstance(other, UniformMarginal):
            return False
        return self.lower() == other.lower() and self.upper() == other.upper()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"UniformMarginal(lower={self.lower()}, upper={self.upper()})"
