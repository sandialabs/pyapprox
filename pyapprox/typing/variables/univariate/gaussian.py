from typing import Tuple, Generic, Any
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class GaussianMarginal(Generic[Array]):
    """
    Gaussian (normal) distribution.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    stdev : float
        Standard deviation of the Gaussian distribution.
    backend : BackendMixin, optional
        The backend to use for computations. Defaults to NumpyMixin.
    """

    def __init__(self, mean: float, stdev: float, bkd: Backend[Array]):
        validate_backend(bkd)
        self._bkd = bkd
        self._mean = mean
        self._stdev = stdev
        self._var = stdev**2
        self._log_const = self._bkd.log(
            1.0
            / (
                math.sqrt((2.0 * math.pi))
                * self._bkd.asarray([self._stdev])[0]
            )
        )

        # Store name and shapes consistent with scipy.stats to allow
        # classes to be used interchangeably
        self._name = "norm"
        self._shapes = None
        self._scales = {
            "loc": self._bkd.array([self._mean]),
            "scale": self._bkd.array([self._stdev]),
        }

    def _transform_scale_parameters(self) -> Tuple[float, float]:
        """
        Transform scale parameters so that when any bounded variable is
        transformed to the canonical domain [-1, 1]
        """
        loc, scale = (
            self._bkd.copy(self._scales["loc"])[0],
            self._bkd.copy(self._scales["scale"])[0],
        )
        return loc, scale

    def is_bounded(self) -> bool:
        """
        Check if the Gaussian distribution is bounded.

        Returns
        -------
        is_bounded : bool
            False for Gaussian distribution (unbounded).
        """
        return False

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF.

        Returns
        -------
        pdf_vals : Array
            The evaluated PDF values.
        """
        return self._bkd.exp(self.logpdf(samples))

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log of the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PDF.

        Returns
        -------
        logpdf_vals : Array
            The evaluated log PDF values.
        """
        return self._log_const - (samples - self._mean) ** 2 / (
            2.0 * self._var
        )

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function (CDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF.

        Returns
        -------
        cdf_vals : Array
            The evaluated CDF values.
        """
        return 0.5 * (
            1.0
            + self._bkd.erf(
                (samples - self._mean) / (self._stdev * math.sqrt(2))
            )
        )

    def ppf(self, usamples: Array) -> Array:
        """
        Evaluate the percent point function (inverse CDF).

        Parameters
        ----------
        usamples : Array
            Points at which to evaluate the inverse CDF.

        Returns
        -------
        ppf_vals : Array
            The evaluated inverse CDF values.
        """
        return (
            math.sqrt(2.0) * self._bkd.erfinv(2.0 * usamples - 1.0)
        ) * self._stdev + self._mean

    def jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        jacobian : Array
            The Jacobian of the PDF.
        """
        # Compute the PDF values
        pdf_vals = self(samples)

        # Compute the log PDF Jacobian
        logpdf_jacobian_vals = self.logpdf_jacobian(samples)

        # Use the chain rule: PDF' = PDF * logPDF'
        return self._bkd.reshape(pdf_vals * logpdf_jacobian_vals, (1, -1))

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.

        Returns
        -------
        jacobian : Array
            The Jacobian of the log PDF.
        """
        return self._bkd.reshape(
            (-(samples - self._mean) / self._var), (1, -1)
        )

    def mean(self) -> float:
        """
        Compute the mean of the Gaussian distribution.

        Returns
        -------
        mean : float
            The mean of the Gaussian distribution.
        """
        return self._mean

    def median(self) -> float:
        """
        Compute the median of the Gaussian distribution.

        Returns
        -------
        median : float
            The median of the Gaussian distribution.
        """
        return self._mean

    def var(self) -> float:
        """
        Compute the variance of the Gaussian distribution.

        Returns
        -------
        var : float
            The variance of the Gaussian distribution.
        """
        return self._var

    def std(self) -> float:
        """
        Compute the standard deviation of the Gaussian distribution.

        Returns
        -------
        std : float
            The standard deviation of the Gaussian distribution.
        """
        return self._stdev

    def __eq__(self, other: Any) -> bool:
        """
        Check if two Gaussian distributions are equal.

        Parameters
        ----------
        other : Marginal
            The other Gaussian distribution.

        Returns
        -------
        equal : bool
            True if the distributions are equal, False otherwise.
        """
        if not isinstance(other, GaussianMarginal):
            return False
        if self._mean != other._mean or self._stdev != other._stdev:
            return False
        return True

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the Gaussian distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Random samples from the Gaussian distribution.
        """
        usamples = self._bkd.asarray(np.random.uniform(0, 1, nsamples))
        return self.ppf(usamples)

    def interval(self, alpha: float) -> Array:
        """
        Compute the interval with a given probability content alpha.

        Parameters
        ----------
        alpha : float
            The probability content of the interval.

        Returns
        -------
        interval : Array
            The interval, represented as a pair of values.

        Notes
        -----
        The interval is calculated using the formula for the interval with a
        given probability content, based on the quantile function ppf.
        """
        eps = (1.0 - alpha) / 2.0
        return self.ppf(self._bkd.array([eps, 1 - eps]))

    def __repr__(self) -> str:
        """
        Return a string representation of the Gaussian distribution.

        Returns
        -------
        repr : str
            A string representation of the Gaussian distribution.
        """
        return "{0}(mean={1}, stdev={2})".format(
            self.__class__.__name__, self._mean, self._stdev
        )
