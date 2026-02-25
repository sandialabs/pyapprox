"""
Full-rank Laplace posterior approximation.

The Laplace approximation computes a Gaussian approximation to the
posterior by using the Hessian of the negative log-posterior at the
MAP point as the precision matrix.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian


class DenseLaplacePosterior(Generic[Array]):
    r"""
    Dense (full-rank) Laplace approximation to the posterior.

    The Laplace approximation approximates the posterior as Gaussian:

    .. math::
        p(x|y) \approx N(x_{MAP}, H^{-1})

    where:
    - :math:`x_{MAP}` is the maximum a posteriori (MAP) estimate
    - :math:`H = -\nabla^2 \log p(x|y)|_{x=x_{MAP}}` is the Hessian of the
      negative log-posterior at the MAP point

    The Hessian can be decomposed as:

    .. math::
        H = H_{prior} + H_{likelihood}

    where :math:`H_{prior}` is the prior precision and :math:`H_{likelihood}`
    is the Hessian of the negative log-likelihood.

    Parameters
    ----------
    map_point : Array
        The MAP point. Shape: (nvars, 1)
    prior_precision : Array
        The prior precision (inverse covariance). Shape: (nvars, nvars)
    likelihood_hessian : Array
        The Hessian of the negative log-likelihood at the MAP point.
        Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> map_point = np.array([[1.0], [2.0]])
    >>> prior_precision = np.eye(2)  # Unit prior precision
    >>> likelihood_hessian = np.array([[2.0, 0.5], [0.5, 1.0]])
    >>> laplace = DenseLaplacePosterior(
    ...     map_point, prior_precision, likelihood_hessian, bkd
    ... )
    >>> laplace.compute()
    >>> posterior_mean = laplace.posterior_mean()
    >>> posterior_cov = laplace.posterior_covariance()
    """

    def __init__(
        self,
        map_point: Array,
        prior_precision: Array,
        likelihood_hessian: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._nvars = map_point.shape[0]

        if map_point.shape != (self._nvars, 1):
            raise ValueError(
                f"map_point has wrong shape {map_point.shape}, "
                f"expected ({self._nvars}, 1)"
            )
        self._map_point = map_point

        if prior_precision.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"prior_precision has wrong shape {prior_precision.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._prior_precision = prior_precision

        if likelihood_hessian.shape != (self._nvars, self._nvars):
            raise ValueError(
                f"likelihood_hessian has wrong shape {likelihood_hessian.shape}, "
                f"expected ({self._nvars}, {self._nvars})"
            )
        self._likelihood_hessian = likelihood_hessian

        # State
        self._posterior_cov: Optional[Array] = None
        self._computed = False

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def compute(self) -> None:
        """
        Compute the Laplace approximation.

        This computes the posterior covariance as the inverse of the
        total Hessian (prior precision + likelihood Hessian).
        """
        # Total Hessian = prior precision + likelihood Hessian
        total_hessian = self._prior_precision + self._likelihood_hessian

        # Posterior covariance = inverse of total Hessian
        self._posterior_cov = self._bkd.inv(total_hessian)
        self._computed = True

    def posterior_mean(self) -> Array:
        """
        Return the posterior mean (MAP point).

        Returns
        -------
        Array
            Posterior mean. Shape: (nvars, 1)
        """
        return self._map_point

    def posterior_covariance(self) -> Array:
        """
        Return the posterior covariance.

        Returns
        -------
        Array
            Posterior covariance. Shape: (nvars, nvars)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return self._posterior_cov

    def covariance_diagonal(self) -> Array:
        """
        Return the diagonal of the posterior covariance (marginal variances).

        Returns
        -------
        Array
            Marginal variances. Shape: (nvars,)

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return self._bkd.diag(self._posterior_cov)

    def posterior_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """
        Return the posterior as a Gaussian distribution object.

        Returns
        -------
        DenseCholeskyMultivariateGaussian
            Posterior Gaussian distribution.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if not self._computed:
            raise RuntimeError("Must call compute() first")
        return DenseCholeskyMultivariateGaussian(
            self.posterior_mean(),
            self.posterior_covariance(),
            self._bkd,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DenseLaplacePosterior(nvars={self._nvars})"
