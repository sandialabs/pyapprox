"""
Independent joint distributions.

Provides joint distributions where all marginals are independent.
The joint PDF is the product of marginal PDFs.
"""

from typing import Generic, Sequence, List

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.protocols import MarginalProtocol


class IndependentJoint(Generic[Array]):
    """
    Joint distribution with independent marginals.

    The joint PDF is the product of marginal PDFs:
        p(x_1, ..., x_n) = p_1(x_1) * ... * p_n(x_n)

    The joint log-PDF is the sum of marginal log-PDFs:
        log p(x) = sum_i log p_i(x_i)

    Parameters
    ----------
    marginals : Sequence[MarginalProtocol[Array]]
        List of marginal distributions.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import norm, beta
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import ScipyContinuousMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [
    ...     ScipyContinuousMarginal(norm(0, 1), bkd),
    ...     ScipyContinuousMarginal(beta(2, 5), bkd),
    ... ]
    >>> joint = IndependentJoint(marginals, bkd)
    >>> samples = joint.rvs(100)  # Shape: (2, 100)
    """

    def __init__(
        self,
        marginals: Sequence[MarginalProtocol[Array]],
        bkd: Backend[Array],
    ):
        if len(marginals) == 0:
            raise ValueError("Must provide at least one marginal distribution")
        self._bkd = bkd
        self._marginals = list(marginals)
        self._nvars = len(marginals)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables (dimension of the distribution).
        """
        return self._nvars

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """
        Return the marginal distributions.

        Returns
        -------
        List[MarginalProtocol]
            List of marginal distributions.
        """
        return self._marginals

    def marginal(self, index: int) -> MarginalProtocol[Array]:
        """
        Return a specific marginal distribution.

        Parameters
        ----------
        index : int
            Index of the marginal.

        Returns
        -------
        MarginalProtocol
            The marginal distribution at given index.
        """
        return self._marginals[index]

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the joint distribution.

        Samples each marginal independently and stacks them.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (nvars, nsamples)
        """
        samples_list = []
        for marginal in self._marginals:
            # Each marginal returns shape (1, nsamples)
            marg_samples = marginal.rvs(nsamples)
            # Flatten to (nsamples,) for stacking
            samples_list.append(self._bkd.flatten(marg_samples))
        return self._bkd.stack(samples_list, axis=0)

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the joint log probability density function.

        For independent marginals:
            log p(x) = sum_i log p_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log PDF values. Shape: (nsamples,)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        nsamples = samples.shape[1]
        logpdf_total = self._bkd.zeros((nsamples,))

        for i, marginal in enumerate(self._marginals):
            logpdf_total = logpdf_total + marginal.logpdf(samples[i])

        return logpdf_total

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability density function.

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

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the joint cumulative distribution function.

        For independent marginals:
            F(x) = prod_i F_i(x_i)

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (nsamples,)
        """
        if samples.ndim == 1:
            samples = self._bkd.reshape(samples, (self._nvars, 1))

        nsamples = samples.shape[1]
        cdf_total = self._bkd.ones((nsamples,))

        for i, marginal in enumerate(self._marginals):
            cdf_total = cdf_total * marginal.cdf(samples[i])

        return cdf_total

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF component-wise.

        For independent marginals, applies inverse CDF to each dimension.

        Parameters
        ----------
        probs : Array
            Probability values. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Quantile values. Shape: (nvars, nsamples)
        """
        if probs.ndim == 1:
            probs = self._bkd.reshape(probs, (self._nvars, 1))

        samples_list = []
        for i, marginal in enumerate(self._marginals):
            samples_list.append(marginal.invcdf(probs[i]))

        return self._bkd.stack(samples_list, axis=0)

    def correlation_matrix(self) -> Array:
        """
        Return the correlation matrix.

        For independent marginals, this is the identity matrix.

        Returns
        -------
        Array
            Identity correlation matrix. Shape: (nvars, nvars)
        """
        return self._bkd.eye(self._nvars)

    def mean(self) -> Array:
        """
        Return the mean of the joint distribution.

        Returns
        -------
        Array
            Mean vector. Shape: (nvars,)
        """
        means = []
        for marginal in self._marginals:
            if hasattr(marginal, "mean_value"):
                means.append(marginal.mean_value())
            else:
                # Fallback: use sampling
                samples = marginal.rvs(10000)
                means.append(float(self._bkd.sum(samples) / 10000))
        return self._bkd.asarray(means)

    def variance(self) -> Array:
        """
        Return the variance of each marginal.

        Returns
        -------
        Array
            Variance vector. Shape: (nvars,)
        """
        variances = []
        for marginal in self._marginals:
            if hasattr(marginal, "variance"):
                variances.append(marginal.variance())
            else:
                # Fallback: use sampling
                samples = marginal.rvs(10000)
                mean = float(self._bkd.sum(samples) / 10000)
                variances.append(
                    float(self._bkd.sum((samples - mean) ** 2) / 10000)
                )
        return self._bkd.asarray(variances)

    def covariance(self) -> Array:
        """
        Return the covariance matrix.

        For independent marginals, this is diagonal with variances.

        Returns
        -------
        Array
            Diagonal covariance matrix. Shape: (nvars, nvars)
        """
        return self._bkd.diag(self.variance())

    def is_bounded(self) -> bool:
        """
        Check if all marginals have bounded support.

        Returns
        -------
        bool
            True if all marginals are bounded.
        """
        for marginal in self._marginals:
            if hasattr(marginal, "is_bounded"):
                if not marginal.is_bounded():
                    return False
            else:
                # Assume unbounded if method not available
                return False
        return True

    def bounds(self) -> Array:
        """
        Return the bounds of the joint distribution.

        Returns
        -------
        Array
            Bounds array. Shape: (2, nvars) where [0, :] is lower, [1, :] is upper.
        """
        lower_bounds = []
        upper_bounds = []

        for marginal in self._marginals:
            if hasattr(marginal, "interval"):
                interval = marginal.interval(1.0)
                lower_bounds.append(float(interval[0]))
                upper_bounds.append(float(interval[1]))
            else:
                # Default to unbounded
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)

        return self._bkd.stack(
            [self._bkd.asarray(lower_bounds), self._bkd.asarray(upper_bounds)],
            axis=0,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IndependentJoint(nvars={self._nvars})"
