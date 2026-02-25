"""
SciPy discrete distribution wrapper.

Provides a wrapper for SciPy discrete distributions.
"""

from typing import Any, Dict, Generic, Tuple

import numpy as np
from scipy.stats import _discrete_distns

from pyapprox.util.backends.protocols import Array, Backend


class ScipyDiscreteMarginal(Generic[Array]):
    """
    Wrapper for SciPy discrete distributions.

    Adapts SciPy frozen discrete random variables to a protocol-compatible
    interface.

    Parameters
    ----------
    scipy_rv : rv_discrete_frozen
        A frozen SciPy discrete random variable.
    bkd : Backend[Array]
        The backend to use for computations.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import binom
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> scipy_rv = binom(n=10, p=0.5)
    >>> dist = ScipyDiscreteMarginal(scipy_rv, bkd)
    >>> samples = np.array([[0, 5, 10]])  # Shape: (1, 3)
    >>> pmf_vals = dist(samples)  # Shape: (1, 3)
    """

    def __init__(self, scipy_rv: Any, bkd: Backend[Array]):
        self._bkd = bkd
        self._validate_marginal(scipy_rv)
        self._scipy_rv = scipy_rv
        self._name, self._scales, self._shapes = self._get_distribution_info()

    def _validate_marginal(self, scipy_rv: Any) -> None:
        """Validate that this is a discrete SciPy random variable."""
        if scipy_rv.dist.name not in _discrete_distns._distn_names:
            raise ValueError("marginal is not a discrete SciPy variable")

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

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

    def pmf(self, samples: Array) -> Array:
        """
        Evaluate the probability mass function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PMF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            PMF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        result = self._bkd.asarray(self._scipy_rv.pmf(self._bkd.to_numpy(samples_1d)))
        return self._bkd.reshape(result, (1, -1))

    def pdf(self, samples: Array) -> Array:
        """Alias for pmf() (protocol compatibility)."""
        return self.pmf(samples)

    def __call__(self, samples: Array) -> Array:
        """Evaluate the PDF/PMF (alias for pdf()/pmf())."""
        return self.pmf(samples)

    def logpmf(self, samples: Array) -> Array:
        """
        Evaluate the log probability mass function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PMF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Log PMF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        result = self._bkd.asarray(
            self._scipy_rv.logpmf(self._bkd.to_numpy(samples_1d))
        )
        return self._bkd.reshape(result, (1, -1))

    # Alias for protocol compatibility
    def logpdf(self, samples: Array) -> Array:
        """Alias for logpmf for protocol compatibility."""
        return self.logpmf(samples)

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
        result = self._bkd.asarray(self._scipy_rv.cdf(self._bkd.to_numpy(samples_1d)))
        return self._bkd.reshape(result, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        For discrete distributions, returns the smallest integer k
        such that CDF(k) >= p.

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Quantile values (integers). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        result = self._bkd.asarray(self._scipy_rv.ppf(self._bkd.to_numpy(probs_1d)))
        return self._bkd.reshape(result, (1, -1))

    # Alias for compatibility
    ppf = invcdf

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
        samples = self._bkd.asarray(self._scipy_rv.rvs(int(nsamples)))
        return self._bkd.reshape(samples, (1, nsamples))

    def mean_value(self) -> float:
        """Return the mean of the distribution."""
        return float(self._scipy_rv.mean())

    def variance(self) -> float:
        """Return the variance of the distribution."""
        return float(self._scipy_rv.var())

    def std(self) -> float:
        """Return the standard deviation."""
        return float(self._scipy_rv.std())

    def median(self) -> float:
        """Return the median of the distribution."""
        return float(self._scipy_rv.median())

    def is_bounded(self) -> bool:
        """
        Check if the distribution has bounded support.

        Returns
        -------
        bool
            True if both endpoints are finite.
        """
        interval = self._scipy_rv.interval(1)
        return bool(np.isfinite(interval[0]) and np.isfinite(interval[1]))

    def interval(self, alpha: float) -> Array:
        """
        Compute the interval with given probability content.

        Parameters
        ----------
        alpha : float
            Probability content of the interval.

        Returns
        -------
        Array
            Interval [lower, upper]. Shape: (1, 2)
        """
        result = self._bkd.asarray(self._scipy_rv.interval(alpha))
        return self._bkd.reshape(result, (1, -1))

    def _get_distribution_info(
        self,
    ) -> Tuple[str, Dict[str, Array], Dict[str, Any]]:
        """Extract distribution name, scales, and shapes from SciPy rv."""
        name = self._scipy_rv.dist.name
        shape_names = self._scipy_rv.dist.shapes

        if shape_names is not None:
            shape_names = [n.strip() for n in shape_names.split(",")]
            shape_values = [
                self._scipy_rv.args[ii]
                for ii in range(min(len(self._scipy_rv.args), len(shape_names)))
            ]
            shape_values += [
                self._scipy_rv.kwds[shape_names[ii]]
                for ii in range(len(self._scipy_rv.args), len(shape_names))
            ]
            shapes = dict(zip(shape_names, shape_values))
        else:
            shapes = {}

        # Extract scale parameters (loc, scale)
        scale_values = [
            self._scipy_rv.args[ii]
            for ii in range(len(shapes), len(self._scipy_rv.args))
        ]
        scale_values += [
            self._scipy_rv.kwds[key] for key in self._scipy_rv.kwds if key not in shapes
        ]

        if len(scale_values) == 0:
            scale_values = [0.0, 1.0]
        elif len(scale_values) == 1:
            if "scale" not in self._scipy_rv.kwds:
                scale_values.append(1.0)
            elif "loc" not in self._scipy_rv.kwds:
                scale_values = [0.0] + scale_values

        scales = {
            "loc": self._bkd.asarray([scale_values[0]]),
            "scale": self._bkd.asarray([scale_values[1]]),
        }

        return name, scales, shapes

    @property
    def name(self) -> str:
        """Distribution name."""
        return self._name

    @property
    def shapes(self) -> Dict[str, Any]:
        """Shape parameters."""
        return self._shapes

    @property
    def scales(self) -> Dict[str, Array]:
        """Scale parameters (loc, scale)."""
        return self._scales

    def probability_masses(self) -> Tuple[Array, Array]:
        """Get the support points and probability masses.

        Returns the discrete points where the distribution has non-zero
        probability and their corresponding probability masses.

        Returns
        -------
        xk : Array
            Support points. Shape: (npoints,)
        pk : Array
            Probability masses. Shape: (npoints,)
        """
        # Get support bounds
        a, b = self._scipy_rv.support()

        # For unbounded distributions, use a probability threshold
        if not np.isfinite(a) or not np.isfinite(b):
            # Find bounds that capture most of the probability mass
            eps = 1e-12
            a_bounded = int(self._scipy_rv.ppf(eps)) if not np.isfinite(a) else int(a)
            b_bounded = (
                int(self._scipy_rv.ppf(1 - eps)) if not np.isfinite(b) else int(b)
            )
            xk = np.arange(a_bounded, b_bounded + 1)
        else:
            xk = np.arange(int(a), int(b) + 1)

        pk = self._scipy_rv.pmf(xk)

        # Filter to points with non-zero probability
        nonzero_mask = pk > 0
        xk = xk[nonzero_mask]
        pk = pk[nonzero_mask]

        return self._bkd.asarray(xk), self._bkd.asarray(pk)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ScipyDiscreteMarginal({self._name}, shapes={self._shapes})"
