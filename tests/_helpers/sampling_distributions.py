"""
Test distributions with known analytical properties.

These distributions are used for validating MCMC samplers by comparing
sample statistics to known analytical values.
"""
# TODO: Consider migrating these distribution classes (BananaLogPosterior,
# GaussianMixtureLogPosterior, CorrelatedGaussianLogPosterior) to
# benchmarks/functions/ as proper benchmark functions with FunctionProtocol
# support. They have well-defined analytical properties useful beyond testing.

from typing import Generic, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class BananaLogPosterior(Generic[Array]):
    """
    Banana-shaped (Rosenbrock-like) distribution.

    log p(x) = -0.5 * ((x0 - mu0)^2/s0^2 + (x1 - x0^2 - mu1)^2/s1^2)

    This is a non-Gaussian distribution with banana-shaped contours.
    The marginal distribution of x0 is Gaussian, but the conditional
    distribution of x1|x0 is non-linear.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mu0 : float, default=0.0
        Mean of first variable.
    mu1 : float, default=0.0
        Offset for second variable.
    s0 : float, default=1.0
        Standard deviation of first variable.
    s1 : float, default=1.0
        Standard deviation of second variable conditional on first.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mu0: float = 0.0,
        mu1: float = 0.0,
        s0: float = 1.0,
        s1: float = 1.0,
    ):
        self._bkd = bkd
        self._mu0 = mu0
        self._mu1 = mu1
        self._s0 = s0
        self._s1 = s1

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate log posterior.

        Parameters
        ----------
        samples : Array
            Shape: (2, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples,)
        """
        x0 = samples[0, :]
        x1 = samples[1, :]

        term1 = ((x0 - self._mu0) / self._s0) ** 2
        term2 = ((x1 - x0**2 - self._mu1) / self._s1) ** 2

        return -0.5 * (term1 + term2)

    # TODO: Pyapprox uses jacobian with shape (nqoi, nvars) never gradient.
    # rename function. This class should meet protocol required by
    # samplers for logposterior.
    def gradient(self, sample: Array) -> Array:
        """
        Evaluate gradient of log posterior.

        Parameters
        ----------
        sample : Array
            Shape: (2, 1)

        Returns
        -------
        Array
            Shape: (2, 1)
        """
        x0 = sample[0, 0]
        x1 = sample[1, 0]

        dx0 = -(x0 - self._mu0) / self._s0**2
        dx0 += 2 * x0 * (x1 - x0**2 - self._mu1) / self._s1**2

        dx1 = -(x1 - x0**2 - self._mu1) / self._s1**2

        return self._bkd.asarray(np.array([[dx0], [dx1]], dtype=np.float64))

    def true_mean(self) -> Tuple[float, float]:
        """
        Return true mean of the distribution.

        For the banana distribution:
        E[x0] = mu0
        E[x1] = E[x0^2] + mu1 = s0^2 + mu0^2 + mu1

        Returns
        -------
        tuple
            (mean_x0, mean_x1)
        """
        mean_x0 = self._mu0
        # E[x0^2] = Var(x0) + E[x0]^2 = s0^2 + mu0^2
        mean_x1 = self._s0**2 + self._mu0**2 + self._mu1
        return (mean_x0, mean_x1)


class GaussianMixtureLogPosterior(Generic[Array]):
    """
    Mixture of two Gaussians.

    p(x) = 0.5 * N(x; mu1, sigma^2) + 0.5 * N(x; mu2, sigma^2)

    This is a bimodal distribution useful for testing sampler
    ability to explore multiple modes.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mu1 : float, default=-2.0
        Mean of first component.
    mu2 : float, default=2.0
        Mean of second component.
    sigma : float, default=1.0
        Standard deviation of both components.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mu1: float = -2.0,
        mu2: float = 2.0,
        sigma: float = 1.0,
    ):
        self._bkd = bkd
        self._mu1 = mu1
        self._mu2 = mu2
        self._sigma = sigma
        self._log_norm = -0.5 * np.log(2 * np.pi * sigma**2)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate log posterior.

        Parameters
        ----------
        samples : Array
            Shape: (1, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples,)
        """
        x = samples[0, :]

        # Log of mixture: log(0.5 * exp(log_p1) + 0.5 * exp(log_p2))
        log_p1 = self._log_norm - 0.5 * ((x - self._mu1) / self._sigma) ** 2
        log_p2 = self._log_norm - 0.5 * ((x - self._mu2) / self._sigma) ** 2

        # Numerically stable log-sum-exp (use bkd to avoid
        # numpy __array_wrap__ deprecation with torch tensors)
        bkd = self._bkd
        log_half = bkd.log(bkd.asarray(0.5))
        max_log = bkd.maximum(log_p1, log_p2)
        result = max_log + bkd.log(
            bkd.exp(log_p1 - max_log) + bkd.exp(log_p2 - max_log)
        )
        return result + log_half

    def true_mean(self) -> float:
        """
        Return true mean of the distribution.

        For equal-weight mixture: E[x] = 0.5 * mu1 + 0.5 * mu2

        Returns
        -------
        float
            Mean of mixture.
        """
        return 0.5 * self._mu1 + 0.5 * self._mu2

    def true_variance(self) -> float:
        """
        Return true variance of the distribution.

        For equal-weight mixture:
        Var(x) = 0.5*(sigma^2 + mu1^2) + 0.5*(sigma^2 + mu2^2) - mean^2
               = sigma^2 + 0.5*(mu1^2 + mu2^2) - mean^2

        Returns
        -------
        float
            Variance of mixture.
        """
        mean = self.true_mean()
        second_moment = self._sigma**2 + 0.5 * (self._mu1**2 + self._mu2**2)
        return second_moment - mean**2


class CorrelatedGaussianLogPosterior(Generic[Array]):
    """
    Multivariate Gaussian with correlation.

    p(x) = N(x; mean, cov)

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mean : Array
        Mean vector. Shape: (nvars,)
    cov : Array
        Covariance matrix. Shape: (nvars, nvars)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mean: Array,
        cov: Array,
    ):
        self._bkd = bkd
        mean_np = bkd.to_numpy(mean).flatten()
        cov_np = bkd.to_numpy(cov)

        self._mean = mean_np
        self._cov = cov_np
        self._nvars = len(mean_np)

        # Precompute for efficiency
        self._prec = np.linalg.inv(cov_np)
        sign, logdet = np.linalg.slogdet(cov_np)
        self._log_norm = -0.5 * (self._nvars * np.log(2 * np.pi) + logdet)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate log posterior.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples,)
        """
        samples_np = self._bkd.to_numpy(samples)
        residuals = samples_np - self._mean[:, None]

        # Mahalanobis distance: sum_i sum_j (x-mu)_i * prec_ij * (x-mu)_j
        quad = np.sum(residuals * (self._prec @ residuals), axis=0)

        return self._bkd.asarray(self._log_norm - 0.5 * quad)

    def gradient(self, sample: Array) -> Array:
        """
        Evaluate gradient of log posterior.

        Parameters
        ----------
        sample : Array
            Shape: (nvars, 1)

        Returns
        -------
        Array
            Shape: (nvars, 1)
        """
        sample_np = self._bkd.to_numpy(sample)
        residual = sample_np - self._mean[:, None]
        grad = -self._prec @ residual
        return self._bkd.asarray(grad.astype(np.float64))

    def true_mean(self) -> np.ndarray:
        """Return true mean."""
        return self._mean.copy()

    def true_covariance(self) -> np.ndarray:
        """Return true covariance."""
        return self._cov.copy()
