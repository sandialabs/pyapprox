"""
Gaussian (normal) univariate distribution.

Provides an analytically-defined Gaussian distribution that implements
MarginalWithJacobianProtocol.
"""

from typing import Generic, Any
import math

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    LogHyperParameter,
    HyperParameterList,
)


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
    >>> samples = np.array([[0.0, 1.0, -1.0]])  # Shape: (1, 3)
    >>> pdf_vals = dist(samples)  # PDF values, shape: (1, 3)
    >>> cdf_vals = dist.cdf(samples)  # CDF values, shape: (1, 3)
    """

    def __init__(self, mean: float, stdev: float, bkd: Backend[Array]):
        self._bkd = bkd

        # Create hyperparameter list for parameter optimization
        # mean: unbounded, no log transform
        self._mean_hyp = HyperParameter(
            name="mean",
            nparams=1,
            values=mean,
            bounds=(-1e10, 1e10),  # Effectively unbounded
            bkd=bkd,
        )
        # stdev: positive, use log transform
        self._stdev_hyp = LogHyperParameter(
            name="stdev",
            nparams=1,
            user_values=stdev,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._mean_hyp, self._stdev_hyp])

        # Constant for log PDF computation
        self._log_2pi = math.log(2.0 * math.pi)

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
        """Return the number of distribution parameters (mean and stdev)."""
        return self._hyp_list.nparams()

    def _get_mean(self) -> Array:
        """Get mean as array (preserves autograd graph)."""
        return self._mean_hyp.get_values()[0]

    def _get_stdev(self) -> Array:
        """Get stdev as array (preserves autograd graph)."""
        return self._stdev_hyp.exp_values()[0]

    def _get_var(self) -> Array:
        """Get variance as array (preserves autograd graph)."""
        stdev = self._get_stdev()
        return stdev * stdev

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            The evaluated PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        return self._bkd.exp(self.logpdf(samples))

    def __call__(self, samples: Array) -> Array:
        """Evaluate the PDF (alias for pdf())."""
        return self.pdf(samples)

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
            The evaluated log PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        mean = self._get_mean()
        stdev = self._get_stdev()
        z = (samples_1d - mean) / stdev
        log_const = -0.5 * self._log_2pi - self._bkd.log(stdev)
        result = log_const - 0.5 * z**2
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
        mean = self._get_mean()
        stdev = self._get_stdev()
        z = (samples_1d - mean) / (stdev * math.sqrt(2.0))
        result = 0.5 * (1.0 + self._bkd.erf(z))
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
            Quantile values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        mean = self._get_mean()
        stdev = self._get_stdev()
        result = (
            math.sqrt(2.0) * self._bkd.erfinv(2.0 * probs_1d - 1.0)
        ) * stdev + mean
        return self._bkd.reshape(result, (1, -1))

    # Alias for compatibility
    ppf = invcdf

    def invcdf_jacobian(self, probs: Array) -> Array:
        """
        Compute Jacobian of inverse CDF.

        d(F^{-1})/dp = 1 / pdf(F^{-1}(p))

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Validate input (also done in invcdf and __call__)
        self._validate_input(probs)
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
            Random samples. Shape: (1, nsamples)
        """
        usamples = self._bkd.asarray(np.random.uniform(0, 1, nsamples))
        # Convert to 2D for invcdf
        usamples_2d = self._bkd.reshape(usamples, (1, nsamples))
        return self.invcdf(usamples_2d)

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        Returns
        -------
        float
            Mean value.
        """
        return float(self._bkd.to_numpy(self._mean_hyp.get_values())[0])

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        Returns
        -------
        float
            Variance value.
        """
        stdev = float(self._bkd.to_numpy(self._stdev_hyp.exp_values())[0])
        return stdev * stdev

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        return float(self._bkd.to_numpy(self._stdev_hyp.exp_values())[0])

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
            Shape: (1, 2)
        """
        eps = (1.0 - alpha) / 2.0
        probs_2d = self._bkd.array([[eps, 1.0 - eps]])  # Shape: (1, 2)
        return self.invcdf(probs_2d)

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        mean = self._get_mean()
        var = self._get_var()
        grad = -(samples_1d - mean) / var
        return self._bkd.reshape(grad, (1, -1))

    def pdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Input validation is done by __call__ and logpdf_jacobian
        pdf_vals = self(samples)
        logpdf_jac = self.logpdf_jacobian(samples)
        return pdf_vals * logpdf_jac

    def logpdf_jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute the Jacobian of log PDF w.r.t. distribution parameters.

        Returns derivatives in the optimizer's parameter space (log-space
        for stdev).

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.
            Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian matrix with shape (nsamples, nparams).
            Column 0: d(logpdf)/d(mean)
            Column 1: d(logpdf)/d(log_stdev)
        """
        samples_1d = self._validate_input(samples)
        mean = self._get_mean()
        stdev = self._get_stdev()

        # z = (x - mean) / stdev
        z = (samples_1d - mean) / stdev

        # d(logpdf)/d(mean) = (x - mean) / stdev^2 = z / stdev
        d_mean = z / stdev

        # d(logpdf)/d(stdev) = -1/stdev + (x-mean)^2/stdev^3 = (-1 + z^2)/stdev
        # d(logpdf)/d(log_stdev) = stdev * d(logpdf)/d(stdev) = -1 + z^2
        d_log_stdev = -1.0 + z**2

        # Stack columns: shape (nsamples, 2)
        return self._bkd.stack([d_mean, d_log_stdev], axis=1)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another GaussianMarginal."""
        if not isinstance(other, GaussianMarginal):
            return False
        return (
            self.mean_value() == other.mean_value()
            and self.std() == other.std()
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianMarginal(mean={self.mean_value()}, stdev={self.std()})"
