"""
Discrete univariate distributions.

Provides discrete distributions that implement MarginalProtocol.
"""

from typing import Generic, Any, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class CustomDiscreteMarginal(Generic[Array]):
    """
    Custom discrete distribution defined by explicit probability masses.

    Implements MarginalProtocol for discrete distributions where the user
    specifies the locations and probabilities of each mass point.

    Parameters
    ----------
    xk : Array
        Locations of discrete probability masses (1D array).
    pk : Array
        Probabilities at each location (1D array, must sum to 1).
    bkd : Backend[Array]
        The backend to use for computations.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> xk = np.array([0.0, 1.0, 2.0])
    >>> pk = np.array([0.2, 0.5, 0.3])
    >>> dist = CustomDiscreteMarginal(xk, pk, bkd)
    >>> dist.pmf(np.array([[0.0, 1.0]]))  # Returns [[0.2, 0.5]], shape (1, 2)
    """

    def __init__(
        self,
        xk: Array,
        pk: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd

        # Convert to backend arrays
        xk_arr = self._bkd.asarray(xk)
        pk_arr = self._bkd.asarray(pk)

        # Validate inputs
        if xk_arr.ndim != 1:
            raise ValueError(f"xk must be 1D array, got shape {xk_arr.shape}")
        if pk_arr.ndim != 1:
            raise ValueError(f"pk must be 1D array, got shape {pk_arr.shape}")
        if len(xk_arr) != len(pk_arr):
            raise ValueError(
                f"xk and pk must have same length, got {len(xk_arr)} and {len(pk_arr)}"
            )
        if len(xk_arr) == 0:
            raise ValueError("xk and pk must have at least one element")

        # Validate probabilities sum to 1
        pk_sum = float(self._bkd.sum(pk_arr))
        if not self._bkd.allclose(
            self._bkd.asarray([pk_sum]),
            self._bkd.asarray([1.0]),
            atol=1e-10,
        ):
            raise ValueError(f"pk must sum to 1, got {pk_sum}")

        # Validate all probabilities are non-negative
        pk_np = self._bkd.to_numpy(pk_arr)
        if np.any(pk_np < 0):
            raise ValueError("All probabilities must be non-negative")

        # Sort by xk values
        xk_np = self._bkd.to_numpy(xk_arr)
        sort_idx = np.argsort(xk_np)
        self._xk = self._bkd.asarray(xk_np[sort_idx])
        self._pk = self._bkd.asarray(pk_np[sort_idx])

        # Compute ECDF (cumulative sum of sorted probabilities)
        self._ecdf = self._bkd.asarray(np.cumsum(pk_np[sort_idx]))

        # Store numpy versions for fast lookups
        self._xk_np = self._bkd.to_numpy(self._xk)
        self._pk_np = self._bkd.to_numpy(self._pk)
        self._ecdf_np = self._bkd.to_numpy(self._ecdf)

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

    def nmasses(self) -> int:
        """Return the number of probability masses."""
        return len(self._xk_np)

    def probability_masses(self) -> Tuple[Array, Array]:
        """
        Return the probability mass locations and values.

        Returns
        -------
        Tuple[Array, Array]
            Tuple of (locations, probabilities).
        """
        return self._xk, self._pk

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability mass function (PMF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PMF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            PMF values (0 for points not at mass locations). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        return self.pmf(samples)

    def pmf(self, samples: Array) -> Array:
        """
        Evaluate the probability mass function (PMF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PMF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            PMF values (0 for points not at mass locations). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        samples_np = self._bkd.to_numpy(samples_1d)
        result = np.zeros(len(samples_np))

        # Tolerance for floating-point comparison
        tol = np.finfo(float).eps * 100

        for ii, x in enumerate(samples_np):
            # Find matching mass point
            matches = np.abs(self._xk_np - x) < tol
            if np.any(matches):
                idx = np.where(matches)[0][0]
                result[ii] = self._pk_np[idx]

        return self._bkd.reshape(self._bkd.asarray(result), (1, -1))

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
            Log PMF values (-inf for points not at mass locations). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Validation done by pmf
        pmf_vals = self.pmf(samples)
        return self._bkd.log(pmf_vals)

    # Aliases for protocol compatibility
    def pdf(self, samples: Array) -> Array:
        """Alias for pmf (protocol compatibility)."""
        return self.pmf(samples)

    def logpdf(self, samples: Array) -> Array:
        """Alias for logpmf (protocol compatibility)."""
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
        samples_np = self._bkd.to_numpy(samples_1d)
        result = np.zeros(len(samples_np))

        for ii, x in enumerate(samples_np):
            if x < self._xk_np[0]:
                result[ii] = 0.0
            elif x >= self._xk_np[-1]:
                result[ii] = 1.0
            else:
                # Find largest xk <= x
                idx = np.searchsorted(self._xk_np, x, side="right") - 1
                if idx >= 0:
                    result[ii] = self._ecdf_np[idx]

        return self._bkd.reshape(self._bkd.asarray(result), (1, -1))

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
            Quantile values (mass point locations). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        probs_np = self._bkd.to_numpy(probs_1d)
        result = np.zeros(len(probs_np))

        for ii, p in enumerate(probs_np):
            if p <= 0:
                result[ii] = self._xk_np[0]
            elif p >= 1:
                result[ii] = self._xk_np[-1]
            else:
                # Find smallest xk such that ecdf >= p
                idx = np.searchsorted(self._ecdf_np, p, side="left")
                result[ii] = self._xk_np[idx]

        return self._bkd.reshape(self._bkd.asarray(result), (1, -1))

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
        samples = np.random.choice(self._xk_np, size=nsamples, p=self._pk_np)
        return self._bkd.reshape(self._bkd.asarray(samples), (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        mean = sum(xk * pk)

        Returns
        -------
        float
            Mean value.
        """
        return float(np.sum(self._xk_np * self._pk_np))

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = sum(xk^2 * pk) - mean^2

        Returns
        -------
        float
            Variance value.
        """
        mean = self.mean_value()
        second_moment = float(np.sum(self._xk_np**2 * self._pk_np))
        return second_moment - mean**2

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        return float(np.sqrt(self.variance()))

    def moment(self, power: int) -> float:
        """
        Compute raw moment of given power.

        E[X^power] = sum(xk^power * pk)

        Parameters
        ----------
        power : int
            Power of the moment.

        Returns
        -------
        float
            Moment value.
        """
        return float(np.sum(self._xk_np**power * self._pk_np))

    def is_bounded(self) -> bool:
        """
        Check if the distribution is bounded.

        Returns
        -------
        bool
            True for discrete distributions (always bounded).
        """
        return True

    def bounds(self) -> Tuple[float, float]:
        """
        Return the support bounds.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds (min and max of mass locations).
        """
        return (float(self._xk_np[0]), float(self._xk_np[-1]))

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
            Interval [lower, upper] such that P(lower <= X <= upper) >= alpha.
            Shape: (1, 2)
        """
        eps = (1.0 - alpha) / 2.0
        probs_2d = self._bkd.array([[eps, 1.0 - eps]])  # Shape: (1, 2)
        return self.invcdf(probs_2d)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another CustomDiscreteMarginal."""
        if not isinstance(other, CustomDiscreteMarginal):
            return False
        if len(self._xk_np) != len(other._xk_np):
            return False
        return bool(
            np.allclose(self._xk_np, other._xk_np)
            and np.allclose(self._pk_np, other._pk_np)
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CustomDiscreteMarginal(nmasses={self.nmasses()})"


class DiscreteChebyshevMarginal(CustomDiscreteMarginal[Array]):
    """
    Uniform discrete distribution over equally-spaced points.

    This is a specialized discrete distribution with uniform probabilities
    over the points {0, 1, 2, ..., nmasses-1}.

    Parameters
    ----------
    nmasses : int
        Number of equally-spaced mass points.
    bkd : Backend[Array]
        The backend to use for computations.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> dist = DiscreteChebyshevMarginal(5, bkd)
    >>> # Uniform distribution over {0, 1, 2, 3, 4}
    >>> dist.pmf(bkd.asarray([[0.0, 2.0, 4.0]]))  # Returns [[0.2, 0.2, 0.2]]
    """

    def __init__(self, nmasses: int, bkd: Backend[Array]):
        if nmasses < 1:
            raise ValueError(f"nmasses must be at least 1, got {nmasses}")

        self._nmasses_init = nmasses

        # Create uniform distribution over {0, 1, ..., nmasses-1}
        xk = np.arange(nmasses, dtype=float)
        pk = np.ones(nmasses) / nmasses

        super().__init__(xk, pk, bkd)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiscreteChebyshevMarginal(nmasses={self._nmasses_init})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another DiscreteChebyshevMarginal."""
        if not isinstance(other, DiscreteChebyshevMarginal):
            return False
        return self._nmasses_init == other._nmasses_init
